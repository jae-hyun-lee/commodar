import time;
import xml.etree.ElementTree as ET
import os;
import shutil;
import glob;
from multiprocessing import Process;
init_time = time.time();


def measure():
	global init_time
	after_time = time.time()
	dif_time = after_time - init_time
	hour = int(dif_time / 3660)
	mins = int((dif_time - hour * 3660) / 60)
	sec = dif_time - hour * 3660 - mins * 60
	print('Processing Time:' + str(hour) + 'hour ' + str(mins) + 'min ' + str(sec) + 'sec ')


def stringClensing(string):
	string = string.replace('\n', '');
	string = string.replace('\"', '');
	string = string.replace('\r', '');
	string = string.strip();
	# string = string.lower();
	return string;


def writeOutput(listString, strOutputName):
	manipulatedData = open(strOutputName, 'w+');
	strNewRow = '\n'.join(listString);
	manipulatedData.write(strNewRow);
	manipulatedData.close();


def PC_event2relation():
	strInputPath = '/data4/jaeh/context/Resources/PC'
	listInputPath = glob.glob(strInputPath + '/*_labeled.xml')

	listWrite = []
	setIncrease = {'Activation', 'Gene_expression', 'Positive_regulation', 'Transcription', 'Translation'}
	setDecrease = {'Inactivation', 'Degradation', 'Negative_regulation'}
	setRelationType = setIncrease | setDecrease | {'Binding', 'Regulation'}

	for strInputFile in listInputPath:
		print(strInputFile)
		doc = ET.parse(strInputFile)
		root = doc.getroot()
		strCorpus = root.get("source")
		for document in root.findall("document"):
			strPMID = document.get("origId").replace("PMID-", "")
			strID = document.get("id")
			strSet = strCorpus + "_" + document.get("set")
			for sentence in document.findall("sentence"):
				strSentenceID = strPMID + sentence.get("id").replace(strID, "")

				dicEntity = {}
				dicTrigger = {}
				listInteraction = []
				dicRelation = {}

				for entity in sentence.findall("entity"):
					if entity.get("given") == "True":
						# {entity id : [charOffset, type]}
						# dicEntity[entity.get("id")] = [entity.get("charOffset"), entity.get("type")]
						# {entity id : charOffset}
						dicEntity[entity.get("id")] = entity.get("charOffset")
					elif entity.get("event") == "True":
						# {triger id : triger_type}
						dicTrigger[entity.get("id")] = entity.get("type")

				for elemInteraction in sentence.findall("interaction"):
					if elemInteraction.get("e1") in dicTrigger:
						# [trigger id, argument id, argument type (theme/cause)]
						listInteraction.append([elemInteraction.get("e1"), elemInteraction.get("e2"), elemInteraction.get("type")])
						# {trigger id: [cause, theme, relation type]}
						dicRelation[elemInteraction.get("e1")] = ['', '', dicTrigger[elemInteraction.get("e1")]]

				for interaction in listInteraction:
					if interaction[0] in dicTrigger:
						if interaction[1] in dicEntity:
							# flat structure
							if dicTrigger[interaction[0]] == "Binding":
								if not dicRelation[interaction[0]][0]:
									# if 'cause' is empty,
									dicRelation[interaction[0]][0] = dicEntity[interaction[1]]
							if interaction[2] == "Cause":
								dicRelation[interaction[0]][0] = dicEntity[interaction[1]]
							elif interaction[2] == "Theme":
								dicRelation[interaction[0]][1] = dicEntity[interaction[1]]

						else:
							# nested structure
							for interaction1 in listInteraction:
								if interaction[1] == interaction1[0]:
									if interaction1[1] in dicEntity:
										# first-ordered nested structure
										if (interaction[2] == "Cause") & (interaction1[2] == "Cause"):
											dicRelation[interaction[0]][0] = dicEntity[interaction1[1]] + " " + dicTrigger[interaction[1]]
										elif (interaction[2] == "Theme") & (interaction1[2] == "Theme"):
											dicRelation[interaction[0]][1] = dicEntity[interaction1[1]] + " " + dicTrigger[interaction[1]]
									else:
										# second-ordered nested structure
										for interaction2 in listInteraction:
											if interaction1[1] == interaction2[0]:
												if interaction2[1] in dicEntity:
													if (interaction[2] == "Cause") & (interaction1[2] == "Cause") & (interaction2[2] == "Cause"):
														dicRelation[interaction[0]][0] = dicEntity[interaction2[1]] + " " + dicTrigger[interaction1[0]] + " " + dicTrigger[interaction2[0]]
													elif (interaction[2] == "Theme") & (interaction1[2] == "Theme") & (interaction2[2] == "Theme"):
														dicRelation[interaction[0]][1] = dicEntity[interaction2[1]] + " " + dicTrigger[interaction1[0]] + " " + dicTrigger[interaction2[0]]

				for listRelation in dicRelation.values():
					if listRelation[0]:
						if listRelation[1]:
							if listRelation[2] in setRelationType:
								# determining the final relation type
								intCause = 1
								intTheme = 1
								intTrigger = 1

								# nested strcutre
								if (' ' in listRelation[0]) | (' ' in listRelation[1]):
									listCause = listRelation[0].split(' ')[1:]
									listTheme = listRelation[1].split(' ')[1:]

									for strCause in listCause:
										if strCause in setIncrease:
											intCause *= 1
										elif strCause in setDecrease:
											intCause *= -1
										else:
											intCause *= 0

									for strTheme in listTheme:
										if strTheme in setIncrease:
											intTheme *= 1
										elif strTheme in setDecrease:
											intTheme *= -1
										else:
											intTheme *= 0

								# flat structure
								if listRelation[2] in setIncrease:
									intTrigger *= 1

								elif listRelation[2] in setDecrease:
									intTrigger *= -1
								else:
									intTrigger *= 0

								intFinalType = intCause * intTheme * intTrigger
								# print(listRelation, intCause, intTheme, intTrigger, intFinalType)

								if listRelation[2] == "Binding":
									listRelation.append("Binding")
								elif intFinalType == 1:
									listRelation.append("Increase")
								elif intFinalType == -1:
									listRelation.append("Decrease")
								else:
									listRelation.append("Regulation")

								listWrite.append('\t'.join([strSet, strSentenceID] + listRelation))

	writeOutput(listWrite, '/data4/jaeh/context/Resources/PC/labeled.tsv')


def GE_membership_check():
	strInputPath = '/data4/jaeh/context/Resources/GE'
	listInputPath = glob.glob(strInputPath + '/*_label.xml')
	dicCorpus = {}

	for strInputFile in listInputPath:
		setDocument = set([])
		doc = ET.parse(strInputFile)
		root = doc.getroot()
		strCorpus = root.get("source")
		for document in root.findall("document"):
			strSet = strCorpus + "_" + document.get("set")
			setDocument.add(document.get("origId").replace("PMID-", ""))

		dicCorpus[strSet] = setDocument
		print(strSet, len(setDocument))
	# print(dicCorpus["GE09_train"])
	print(len(dicCorpus["GE09_train"] - dicCorpus["GE11_train"]),
		  len(dicCorpus["GE11_train"] - dicCorpus["GE09_train"]))
	print(len(dicCorpus["GE09_devel"] - dicCorpus["GE11_devel"]),
		  len(dicCorpus["GE11_devel"] - dicCorpus["GE09_devel"]))
	print(len(dicCorpus["GE11_train"] - dicCorpus["GE13_train"]),
		  len(dicCorpus["GE13_train"] - dicCorpus["GE11_train"]))
	print(len(dicCorpus["GE11_devel"] - dicCorpus["GE13_devel"]),
		  len(dicCorpus["GE13_devel"] - dicCorpus["GE11_devel"]))


def GE_event2relation():

	# reading metamap results
	strInputPath = '/data4/jaeh/context/Resources/GE'
	dicDisease = {}
	with open(strInputPath + "/GE_MetaMapped_disease.tsv", "r") as fileInput:
		fileInput.readline() # header
		for strInstance in fileInput:
			strInstance = stringClensing(strInstance)
			listInstance = strInstance.split("\t")
			if listInstance[1] in dicDisease:
				dicDisease[listInstance[1]].append(listInstance[3:])
			else:
				dicDisease[listInstance[1]] = [listInstance[3:]]

	# listInputPath = glob.glob(strInputPath + '/*_labeled.xml')
	listInputPath = ["/data4/jaeh/context/Resources/GE/GE11-devel_labeled.xml",
					 "/data4/jaeh/context/Resources/GE/GE11-train_labeled.xml",
					 "/data4/jaeh/context/Resources/GE/GE13-devel_labeled.xml",
					 "/data4/jaeh/context/Resources/GE/GE13-train_labeled.xml"]

	listWrite = []
	setIncrease = {'Activation', 'Gene_expression', 'Positive_regulation', 'Transcription', 'Translation'}
	setDecrease = {'Inactivation', 'Degradation', 'Negative_regulation'}
	setRelationType = setIncrease | setDecrease | {'Binding', 'Regulation'}

	setPMID = set([])

	for strInputFile in listInputPath:
		print(strInputFile)
		doc = ET.parse(strInputFile)
		root = doc.getroot()
		strCorpus = root.get("source")
		for document in root.findall("document"):
			strPMID = document.get("origId").replace("PMID-", "")
			if not strPMID in setPMID:
				setPMID.add(strPMID)
				strID = document.get("id")
				strSet = strCorpus + "_" + document.get("set")
				for sentence in document.findall("sentence"):
					strSentenceID = strPMID + sentence.get("id").replace(strID, "")

					if strSentenceID in dicDisease:
						strSentence = sentence.get("text")

						dicEntity = {}
						dicTrigger = {}
						listInteraction = []
						dicRelation = {}

						# building the entity dictionary
						for entity in sentence.findall("entity"):
							if entity.get("given") == "True":
								# {entity id : [charOffset, type]}
								# dicEntity[entity.get("id")] = [entity.get("charOffset"), entity.get("type")]

								# {entity id : charOffset}
								# dicEntity[entity.get("id")] = entity.get("charOffset")

								# {entity id : [charOffset, text, type]}
								dicEntity[entity.get("id")] = [entity.get("charOffset"), entity.get("text"), entity.get("type")]
							elif entity.get("event") == "True":
								# {triger id : triger_type}
								dicTrigger[entity.get("id")] = entity.get("type")

						# determining disease NER failure based on character offsets of entities
						listDiseaseSet = dicDisease[strSentenceID]
						for i in range(len(listDiseaseSet)):
							for listEntity in dicEntity.values():
								if listDiseaseSet[i][0] == listEntity[0]:
									listDiseaseSet[i] = []
									break

						# building the interaction dictionary
						for elemInteraction in sentence.findall("interaction"):
							if elemInteraction.get("e1") in dicTrigger:
								# [trigger id, argument id, argument type (theme/cause)]
								listInteraction.append([elemInteraction.get("e1"), elemInteraction.get("e2"), elemInteraction.get("type")])
								# {trigger id: [cause_id, theme_id, relation type]}
								dicRelation[elemInteraction.get("e1")] = ['', '', dicTrigger[elemInteraction.get("e1")]]

						# determining the terminal entities of each interaction
						for interaction in listInteraction:
							if interaction[0] in dicTrigger:
								if interaction[1] in dicEntity:
									# flat structure
									if dicTrigger[interaction[0]] == "Binding":
										if not dicRelation[interaction[0]][0]:
											# if 'cause' is empty,
											dicRelation[interaction[0]][0] = interaction[1]
									if interaction[2] == "Cause":
										dicRelation[interaction[0]][0] = interaction[1]
									elif interaction[2] == "Theme":
										dicRelation[interaction[0]][1] = interaction[1]

								else:
									# nested structure
									for interaction1 in listInteraction:
										if interaction[1] == interaction1[0]:
											if interaction1[1] in dicEntity:
												# first-ordered nested structure
												if (interaction[2] == "Cause") & (interaction1[2] == "Cause"):
													dicRelation[interaction[0]][0] = interaction1[1] + " " + dicTrigger[interaction[1]]
												elif (interaction[2] == "Theme") & (interaction1[2] == "Theme"):
													dicRelation[interaction[0]][1] = interaction1[1] + " " + dicTrigger[interaction[1]]
											else:
												# second-ordered nested structure
												for interaction2 in listInteraction:
													if interaction1[1] == interaction2[0]:
														if interaction2[1] in dicEntity:
															if (interaction[2] == "Cause") & (interaction1[2] == "Cause") & (interaction2[2] == "Cause"):
																dicRelation[interaction[0]][0] = interaction2[1] + " " + dicTrigger[interaction1[0]] + " " + dicTrigger[interaction2[0]]
															elif (interaction[2] == "Theme") & (interaction1[2] == "Theme") & (interaction2[2] == "Theme"):
																dicRelation[interaction[0]][1] = interaction2[1] + " " + dicTrigger[interaction1[0]] + " " + dicTrigger[interaction2[0]]

						# determining the final relation types and writing
						for listRelation in dicRelation.values():
							if listRelation[0]:
								if listRelation[1]:
									if listRelation[2] in setRelationType:

										# retrieving the entity information to determine the direction
										listEntity1 = dicEntity[listRelation[0].split(" ")[0]]
										listEntity2 = dicEntity[listRelation[1].split(" ")[0]]
										if int(listEntity1[0].split("-")[0]) < int(listEntity2[0].split("-")[0]):
											strDirection = "Forward"
										elif int(listEntity1[0].split("-")[0]) > int(listEntity2[0].split("-")[0]):
											strDirection = "Backward"
										else:
											continue

										listRelationWrite = listEntity1 + listEntity2

										# determining the final relation type
										intCause = 1
										intTheme = 1
										intTrigger = 1

										# nested strcutre
										if (' ' in listRelation[0]) | (' ' in listRelation[1]):


											listCause = listRelation[0].split(' ')[1:]
											listTheme = listRelation[1].split(' ')[1:]

											for strCause in listCause:
												if strCause in setIncrease:
													intCause *= 1
												elif strCause in setDecrease:
													intCause *= -1
												else:
													intCause *= 0

											for strTheme in listTheme:
												if strTheme in setIncrease:
													intTheme *= 1
												elif strTheme in setDecrease:
													intTheme *= -1
												else:
													intTheme *= 0

										# flat structure
										if listRelation[2] in setIncrease:
											intTrigger *= 1

										elif listRelation[2] in setDecrease:
											intTrigger *= -1
										else:
											intTrigger *= 0

										intFinalType = intCause * intTheme * intTrigger
										# print(listRelation, intCause, intTheme, intTrigger, intFinalType)
										if listRelation[2] == "Binding":
											strLabel = "Binding\tNA"
										elif intFinalType == 1:
											strLabel = "Increase\t" + strDirection
										elif intFinalType == -1:
											strLabel = "Decrease\t" + strDirection
										else:
											strLabel = "Regulation\t" + strDirection

										for listDisease in listDiseaseSet:
											if len(listDisease) > 0:
												listWrite.append('\t'.join([strSet, strSentenceID, strSentence] + listRelationWrite + listDisease + [strLabel]))

	listSorted = sorted(list(set(listWrite)))
	intIndex = 20000
	for i in range(len(listSorted)):
		listSorted[i] = str(intIndex) + "\t" + listSorted[i]
		intIndex += 1
	writeOutput(listSorted, '/data4/jaeh/context/Resources/GE/labeled.tsv')


def w2d_arrange():
	setWrite = set()
	strCorpusPath = '/data4/jaeh/context/Resources/PC/PC13_MetaMapped_disease_w2G1D.tsv'
	with open(strCorpusPath, "r") as fileInput:
		for strInstance in fileInput:
			strInstance = stringClensing(strInstance)
			listInstance = strInstance.split("\t")
			strTemp = listInstance[10].replace(" ", "-")
			listInstance[10] = strTemp

			if '-' in listInstance[5]:
				intGene1Offset = int(listInstance[5].split("-")[0])
				intGene2Offset = int(listInstance[8].split("-")[0])

				if intGene1Offset > intGene2Offset:
					listInstance = listInstance[:3] + listInstance[8:9] + listInstance[6:8] + listInstance[5:6] + listInstance[3:5] + listInstance[9:]
				else:
					listInstance = listInstance[:3] + listInstance[5:6] + listInstance[3:5] + listInstance[8:9] + listInstance[6:8] + listInstance[9:]
				setWrite.add("\t".join(listInstance))
	writeOutput(sorted(list(setWrite)), strCorpusPath.replace(".tsv", "_arranged.tsv"))


def labeling():
	strLabelPath = '/data4/jaeh/context/Resources/PC/labeled.tsv'
	dicLabel = {}
	with open(strLabelPath, "r") as fileInput:
		for strInstance in fileInput:
			strInstance = stringClensing(strInstance)
			listInstance = strInstance.split("\t")
			# { sentence ID : [[ cause, theme, type ], [], ... ] }
			if ' ' in listInstance[2]:
				strCause = listInstance[2].split(' ')[0]
			else:
				strCause = listInstance[2]

			if ' ' in listInstance[3]:
				strTheme = listInstance[3].split(' ')[0]
			else:
				strTheme = listInstance[3]

			if listInstance[1] in dicLabel:
				dicLabel[listInstance[1]].append([strCause, strTheme, listInstance[5]])
			else:
				dicLabel[listInstance[1]] = [[strCause, strTheme, listInstance[5]]]

	listWriteTemp = []
	strCorpusPath = '/data4/jaeh/context/Resources/PC/PC13_MetaMapped_disease_w2G1D_arranged.tsv'
	with open(strCorpusPath, "r") as fileInput:
		for strInstance in fileInput:
			strInstance = stringClensing(strInstance)
			listInstance = strInstance.split("\t")

			if listInstance[1] in dicLabel:
				listTemp = []
				for listRelation in dicLabel[listInstance[1]]:
					if ' '.join(sorted(listRelation)[:2]) == ' '.join(sorted([listInstance[3], listInstance[6]])):
						if listRelation[2] == "Binding":
							listTemp.append(strInstance + "\t" + listRelation[2] + "\tNA")
						elif listRelation[0] == listInstance[3]:
							listTemp.append(strInstance + "\t" + listRelation[2] + "\tForward")
						elif listRelation[0] == listInstance[6]:
							listTemp.append(strInstance + "\t" + listRelation[2] + "\tBackward")
						else:
							listTemp.append(strInstance + "\t" + listRelation[2] + "\tUnknown")

				if not len(listTemp):
					listTemp.append(strInstance + "\tNegative\tNA")

				listWriteTemp += listTemp
			
			else:
				listWriteTemp.append(strInstance + "\tNegative\tNA")

	intLine = 0
	listWrite = []
	for strLine in sorted(list(set(listWriteTemp))):
		listWrite.append(str(intLine) + "\t" + strLine)
		intLine += 1
	writeOutput(listWrite, '/data4/jaeh/context/Resources/PC/PC13_w2G1D_labeled.tsv')


if __name__ == '__main__':
	# PC_event2relation()
	GE_event2relation()
	# w2d_arrange()
	# labeling()

	# GE_membership_check()
	measure()

