package banner.eval.dataset;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.configuration.HierarchicalConfiguration;
import org.apache.commons.lang.NotImplementedException;

import banner.tokenization.SimpleTokenizer;
import banner.types.EntityType;
import banner.types.Mention;
import banner.types.Sentence;
import banner.types.Mention.MentionType;
import banner.types.Token;
import banner.util.SentenceBreaker;

public class PubtatorDataset extends Dataset {

	private BufferedReader dataFile;
	private String currentLine;
	private SentenceBreaker sb;

	public PubtatorDataset() {
		sb = new SentenceBreaker();
	}

	@Override
	public void load(HierarchicalConfiguration config) {
		HierarchicalConfiguration localConfig = config.configurationAt(this.getClass().getPackage().getName());
		String dataFilename = localConfig.getString("dataFilename");
		String pmidsFilename = localConfig.getString("pmidsFilename");
		String foldType = localConfig.getString("foldType");

		List<Abstract> abstracts = new ArrayList<Abstract>();
		try {
			Set<String> pmids = null;
			if (pmidsFilename != null) {			
				pmids = getPMIDs(pmidsFilename);
			}
			dataFile = new BufferedReader(new FileReader(dataFilename));
			nextLine();
			Abstract a = getAbstract();
			while (a != null) {
				if (pmids == null || pmids.contains(a.getId()))
					abstracts.add(a);
				a = getAbstract();
			}
			dataFile.close();
			if (pmids != null && pmids.size() != abstracts.size())
				System.out.println("WARNING: pmids.size() = " + pmids.size() + ", abstracts.size() = " + abstracts.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		for (Abstract a : abstracts) {
			int start = 0;
			int end = 0;
			List<Tag> tags = a.getTags();
			List<String> sentenceTexts = a.getSentenceTexts();
			for (int i = 0; i < sentenceTexts.size(); i++) {
				String sentenceText = sentenceTexts.get(i);
				String sentenceId = Integer.toString(i);
				if (sentenceId.length() < 2)
					sentenceId = "0" + sentenceId;
				sentenceId = a.getId() + "-" + sentenceId;
				Sentence s = new Sentence(sentenceId, a.getId(), sentenceText);
				// System.out.println(a.getId() + "\t" + sentenceText);
				tokenizer.tokenize(s);
				// Add mentions
				end += sentenceText.length();
				for (Tag tag : new ArrayList<Tag>(tags)) {
					if (tag.start >= start && tag.end <= end) {
						// TODO Verify that the match between the mentions and the tokens is acceptable
						int tagstart = s.getTokenIndexStart(tag.start - start);
						if (tagstart < 0)
							throw new IllegalArgumentException("Mention start does not match start of token: " + tagstart);
						int tagend = s.getTokenIndexEnd(tag.end - start);
						if (tagend < 0)
							throw new IllegalArgumentException("Mention end does not match end of token: " + tagend);
						tagend += 1; // this side is exclusive
						EntityType type = tag.type;
						if (foldType != null)
							type = EntityType.getType(foldType);
						Mention mention = new Mention(s, tagstart, tagend, type, MentionType.Required);
						if (!mention.getText().equals(sentenceText.substring(tag.start - start, tag.end - start)))
							throw new IllegalArgumentException();
						// System.out.println("\t" + mention.getText() + "\t" + tagstart + "\t" + tagend + "\t" + sentenceText.substring(tag.start - start, tag.end - start));
						Set<String> ids = tag.getIds();
						if (ids.size() > 1)
							throw new IllegalArgumentException();
						for (String conceptId : ids)
							mention.setConceptId(conceptId);
						s.addMention(mention);
						tags.remove(tag);
					}
				}
				// System.out.println();
				sentences.add(s);
				start += sentenceText.length();
			}
			assert tags.size() == 0;
		}
	}

	private class Abstract {
		private String id;
		private List<Tag> tags;
		private List<String> sentenceTexts;

		public Abstract() {
			// Empty
			tags = new ArrayList<Dataset.Tag>();
			sentenceTexts = new ArrayList<String>();
		}

		public String getId() {
			return id;
		}

		public void setId(String id) {
			this.id = id;
		}

		public void setTitleText(String titleText) {
			sentenceTexts.add(titleText + " ");
		}

		public void setAbstractText(String abstractText) {
			sb.setText(abstractText);
			sentenceTexts.addAll(sb.getSentences());
		}

		public List<Tag> getTags() {
			return tags;
		}

		public void addTag(Tag tag) {
			tags.add(tag);
		}

		public String getSubText(int start, int end) {
			for (int i = 0; i < sentenceTexts.size(); i++) {
				String s = sentenceTexts.get(i);
				int length = s.length();
				if (end <= length) {
					return s.substring(start, end);
				}
				start -= s.length();
				end -= s.length();
			}
			return null;
		}

		public List<String> getSentenceTexts() {
			return sentenceTexts;
		}
	}

	private Abstract getAbstract() throws IOException {
		if (currentLine() == null)
			return null;
		Abstract a = new Abstract();
		getTitleText(a);
		getAbstractText(a);
		Tag t = getTag(a);
		while (t != null) {
			a.addTag(t);
			t = getTag(a);
		}
		return a;
	}

	private void getTitleText(Abstract a) throws IOException {
		String[] split = currentLine().split("\\|");
		if (split.length != 3)
			throw new IllegalArgumentException();
		a.setId(split[0]);
		if (!split[1].equals("t"))
			throw new IllegalArgumentException();
		a.setTitleText(split[2]);
		nextLine();
	}

	private void getAbstractText(Abstract a) throws IOException {
		String[] split = currentLine().split("\\|");
		if (split.length != 3)
			throw new IllegalArgumentException();
		if (!split[0].equals(a.getId()))
			throw new IllegalArgumentException();
		if (!split[1].equals("a"))
			throw new IllegalArgumentException();
		a.setAbstractText(split[2]);
		nextLine();
	}

	private Tag getTag(Abstract a) throws IOException {
		String line = currentLine();
		if (line == null)
			return null;
		String[] split = line.split("\t");
		if (split.length != 5 && split.length != 6)
			return null;
		if (!split[0].equals(a.getId()))
			throw new IllegalArgumentException();
		int start = Integer.parseInt(split[1]);
		int end = Integer.parseInt(split[2]);
		String text = a.getSubText(start, end);
		if (!split[3].equals(removePunctuation(text)))
			throw new IllegalArgumentException();
		if (!text.equals(text.trim()))
			throw new IllegalArgumentException();
		String typeText = split[4];
		EntityType type = EntityType.getType(typeText);
		Tag t = new Tag(type, start, end);
		String conceptId = null;
		if (split.length > 5) {
			conceptId = split[5].trim().replaceAll("\\*", "");
			if (conceptId.length() > 0) {
				t.addId(conceptId);
			} else {
				System.out.println("WARNING: " + a.getId() + " lists no concept for annotation \"" + text + "\"");
			}
		} else {
			System.out.println("WARNING: " + a.getId() + " lists no concept for annotation \"" + text + "\"");
		}
		// TODO How to handle + && | ids
		nextLine();
		return t;
	}

	private String removePunctuation(String text) {
		String remove = "\"";
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (remove.indexOf(c) == -1) {
				sb = sb.append(c);
			} else {
				sb.append(" ");
			}
		}
		return sb.toString();
	}

	private String currentLine() throws IOException {
		return currentLine;
	}

	private String nextLine() throws IOException {
		do {
			currentLine = dataFile.readLine();
			if (currentLine != null)
				currentLine = currentLine.trim();
		} while (currentLine != null && currentLine.length() == 0);
		return currentLine;
	}

	private Set<String> getPMIDs(String filename) throws IOException {
		Set<String> pmids = new HashSet<String>();
		BufferedReader dataFile = new BufferedReader(new FileReader(filename));
		String line = dataFile.readLine();
		while (line != null) {
			line = line.trim();
			if (line.length() > 0)
				pmids.add(line);
			line = dataFile.readLine();
		}
		dataFile.close();
		return pmids;
	}

	@Override
	public List<Dataset> split(int n) {
		throw new NotImplementedException();
	}

	public static void main(String[] args) {
		HierarchicalConfiguration config = new HierarchicalConfiguration();
		config.setProperty("banner.eval.dataset.dataFilename", "data/Corpus.txt");
		config.setProperty("banner.eval.dataset.pmidsFilename", "data/NCBI_corpus_training_PMIDs.txt");
		PubtatorDataset d = new PubtatorDataset();
		d.setTokenizer(new SimpleTokenizer());
		d.load(config);

		TObjectIntHashMap<String> counts = new TObjectIntHashMap<String>();
		int total = 0;
		for (Sentence s : d.getSentences()) {
			List<Token> t = s.getTokens();
			total += s.getMentions().size();
			for (Mention m : s.getMentions()) {
				if (m.getEnd() < t.size()) {
					String next = t.get(m.getEnd()).getText().toLowerCase();
					if (counts.contains(next)) {
						counts.increment(next);
					} else {
						counts.put(next, 1);
					}
				}
			}
		}
		System.out.println("Mentions:" + total);
		for (String next : counts.keySet()) {
			System.out.println(next + "\t" + counts.get(next));
		}
	}
}
