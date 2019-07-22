package banner.util;

import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class SentenceBreaker {

	private String text;
	private List<String> sentences;

	public SentenceBreaker() {
		// Empty
	}

	public void setText(String text) {
		this.text = text;
		sentences = new ArrayList<String>();
		BreakIterator bi = BreakIterator.getSentenceInstance(Locale.US);
		bi.setText(text);
		int index = 0;
		int depth = 0;
		while (bi.next() != BreakIterator.DONE) {
			String sentence = text.substring(index, bi.current());
			if (depth > 0) {
				depth += getParenDepth(sentence);
				int last = sentences.size() - 1;
				sentence = sentences.get(last) + sentence;
				sentences.set(last, sentence);
			} else {
				depth += getParenDepth(sentence);
				sentences.add(sentence);
			}
			index = bi.current();
		}
	}

	private int getParenDepth(String text) {
		int depth = 0;
		for (int i = 0; i < text.length(); i++) {
			if (text.charAt(i) == '(')
				depth++;
			if (text.charAt(i) == ')')
				depth--;
		}
		return depth;
	}

	public String getText() {
		return text;
	}

	public List<String> getSentences() {
		return Collections.unmodifiableList(sentences);
	}

	public static void main(String[] args) {
		SentenceBreaker sb = new SentenceBreaker();
		sb.setText("This is short. Testing (A. B. C. E.) also. And another.");
		for (String sentence : sb.getSentences())
			System.out.println(sentence);
	}
}
