package banner.tagging.dictionary;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;
import java.io.FileInputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collections;
import java.util.Arrays;
import java.util.Map;
import java.util.List;
import java.util.Set;

import org.apache.commons.configuration.HierarchicalConfiguration;

import banner.types.EntityType;

public class UMLSMetathesaurusDictionaryTagger_2013AA extends DictionaryTagger {

	public UMLSMetathesaurusDictionaryTagger_2013AA() {
		super();
	}

	@Override
	@SuppressWarnings("unchecked")
	public void load(HierarchicalConfiguration config) throws IOException {
		HierarchicalConfiguration localConfig = config.configurationAt(this.getClass().getName());

		Map<String, EntityType> typeMap = null;
		int maxIndex = localConfig.getMaxIndex("types");
		if (maxIndex >= 0)
			typeMap = new HashMap<String, EntityType>();
		for (int i = 0; i <= maxIndex; i++) {
			Set<String> typeNames = new HashSet<String>(localConfig.getList("types(" + i + ").name"));
			String mappedTypeName = localConfig.getString("types(" + i + ").mapTo");
			EntityType mappedType = null;
			if (mappedTypeName != null) {
				mappedType = EntityType.getType(mappedTypeName);
				for (String typeName : typeNames)
					typeMap.put(typeName, mappedType);
			} else {
				for (String typeName : typeNames)
					typeMap.put(typeName, EntityType.getType(typeName));
			}
		}
		for (String typeName : typeMap.keySet())
			System.out.println("Type name \"" + typeName + "\" becomes \"" + typeMap.get(typeName).getText() + "\"");

		Set<String> allowedLang = null;
		if (localConfig.containsKey("allowedLang"))
			allowedLang = new HashSet<String>(localConfig.getList("allowedLang"));

		Set<String> allowedPref = null;
		if (localConfig.containsKey("allowedPref"))
			allowedPref = new HashSet<String>(localConfig.getList("allowedPref"));

		Set<String> allowedSupp = null;
		if (localConfig.containsKey("allowedSupp"))
			allowedSupp = new HashSet<String>(localConfig.getList("allowedSupp"));

		String semanticTypesFolder = localConfig.getString("dirWithMRSTY");
		Map<String, Set<EntityType>> cuiToTypeMap = loadTypes(semanticTypesFolder, typeMap);
		String conceptNamesFolder = localConfig.getString("dirWithMRCONSO");
		loadConcepts(conceptNamesFolder, cuiToTypeMap, allowedLang, allowedPref, allowedSupp);
	}

	private Map<String, Set<EntityType>> loadTypes(String semanticTypesFolder, Map<String, EntityType> typeMap) throws IOException {
		Map<String, Set<EntityType>> cuiToTypeMap = new HashMap<String, Set<EntityType>>();
		MultiFileReader reader = new MultiFileReader(semanticTypesFolder, "MRSTY.RRF");
		String line = reader.readLine();
		int lineNum = 0;
		LineFieldParser parser = new LineFieldParser();
		while (line != null) {
			parser.init(line);
			String CUI = parser.getField(0);
			String semanticType = parser.getField(1);
			if (typeMap == null || typeMap.containsKey(semanticType)) {
				Set<EntityType> types = cuiToTypeMap.get(CUI);
				if (types == null) {
					types = new HashSet<EntityType>(1);
					cuiToTypeMap.put(CUI, types);
				}
				EntityType type = null;
				if (typeMap == null)
					type = EntityType.getType(semanticType);
				else
					type = typeMap.get(semanticType);
				types.add(type);
			}
			line = reader.readLine();
			lineNum++;
			if (lineNum % 100000 == 0) {
				System.out.println("loadTypes() Line: " + lineNum + " Entries: " + cuiToTypeMap.size() + " types: " + EntityType.getTypes().size());
			}
		}
		reader.close();
		return cuiToTypeMap;
	}

	private void loadConcepts(String conceptNamesFolder, Map<String, Set<EntityType>> cuiToTypeMap, Set<String> allowedLang, Set<String> allowedPref, Set<String> allowedSupp) throws IOException {
		MultiFileReader reader = new MultiFileReader(conceptNamesFolder, "MRCONSO.RRF");
		String line = reader.readLine();
		int lineNum = 0;
		LineFieldParser parser = new LineFieldParser();
		while (line != null) {
			parser.init(line);

			Set<EntityType> types = cuiToTypeMap.get(parser.getField(0)); // CUI
			boolean add = types != null;
			add &= (allowedLang == null) || (allowedLang.contains(parser.getField(1))); // Language
			add &= (allowedPref == null) || (allowedPref.contains(parser.getField(6))); // Preferred
			String name = parser.getField(14);
			add &= (allowedSupp == null) || (allowedSupp.contains(parser.getField(16))); // Suppressed

			if (add) {
				// List<String> tokens = process(name);
				// List<String> tokens2 = new ArrayList<String>();
				// for (String token : tokens)
				// {
				// if (token.matches(("^[A-Za-z0-9]*$")))
				// tokens2.add(token);
				// }
				// add(tokens2, types);
				// System.out.println("Dictionary adding: " + name);
				add(name, types);
			}
			line = reader.readLine();
			lineNum++;
			if (lineNum % 100000 == 0) {
				System.out.println("loadConcepts() Line: " + lineNum + " Entries: " + size() + " types: " + EntityType.getTypes().size());
			}
		}
		reader.close();
	}

	private class MultiFileReader
	{
		private List<File> files;
		private BufferedReader currentReader;
		private int currentFile;
		private String currentLine;
	
		public MultiFileReader(String folder, String filename) throws IOException
		{
			final String filter = filename;
			File dir = new File(folder);
			File[] filesArray = dir.listFiles(new FilenameFilter() {
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith(filter);
				}
			});
			files = Arrays.asList(filesArray);
			Collections.sort(files);
			currentFile = -1;
			incrementFile();
		}
		
		public String readLine() throws IOException
		{
			String nextLine = null;
			if (currentReader != null)
			{
				nextLine = currentReader.readLine();
				if (nextLine == null)
				{
					// Handle end of file
					// TODO This assumes that the last line in a file will always be concatenated with the first line of the next file
					String partialLine = currentLine;
					incrementFile();
					if (currentReader != null)
					{
						currentLine = partialLine + currentLine;
						nextLine = currentReader.readLine();
					}
					else
					{
						currentLine = partialLine + currentLine;
					}
				}
			}
			String returnLine = currentLine;
			currentLine = nextLine;
			return returnLine;
		}
		 
		private void incrementFile() throws IOException
		{
			if (currentReader != null)
			{
				currentReader.close();
				currentReader = null;
			}
			currentFile++;
			if (currentFile < files.size())
			{
				String filename = files.get(currentFile).getCanonicalPath();
				System.out.println("Reading from file " + filename);
				if (filename.endsWith(".gz"))
				{
					currentReader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(filename)), "UTF8"));
				}
				else
				{
					currentReader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF8"));
				}
				currentLine = currentReader.readLine();
			}
		}
		
		public void close() throws IOException
		{
			if (currentReader != null)
			{
				currentReader.close();
				currentReader = null;
			}
			files = null;
			currentLine = null;
		}
	}

	private static class LineFieldParser {
		// Using this class is much faster than using a call to string.split()
		private String line;
		private int currentField;
		private int beginIndex;
		private int endIndex;

		public LineFieldParser() {
			// Empty
		}

		public void init(String line) {
			this.line = line;
			currentField = 0;
			beginIndex = 0;
			endIndex = line.indexOf("|", beginIndex);
		}

		public String getField(int field) {
			if (field < currentField)
				throw new IllegalStateException("Cannot request a field lower than current field");
			while (currentField < field)
				advance();
			return line.substring(beginIndex, endIndex);
		}

		private void advance() {
			beginIndex = endIndex + 1;
			endIndex = line.indexOf("|", beginIndex);
			if (endIndex == -1)
				endIndex = line.length();
			currentField++;
		}
	}
}
