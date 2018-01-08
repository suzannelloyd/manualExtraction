import re, spacy, string, pickle, csv, os, sys
from pprint import pprint
from analyze import analyze_entities, get_native_encoding_type
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.base import TransformerMixin
from nltk.stem.snowball import SnowballStemmer

# Constants denoting various algorithms implemented.
SPACY_ENTITIES_ALGO = "Spacy Entities"
SPACY_NOUN_CHUNKS_ALGO = "Spacy Noun Chunks"
GOOGLE_API_ALGO = "Google Cloud API"

key_delimiter = ';'
spacyParser = spacy.load("en")

#lambda methods
is_not_null_or_integer = lambda s: (not s) or (not s.isdigit() and not (s[0] == '-' and s[1:].isdigit()))

# Creates an easily accessible payload object from a dictionary
class Payload(object):
    def __init__(self, j):
        self.__dict__ = j

# Algo classes

class Algo:
    def getName():
        raise NotImplementedError("Derived classes will implement this")

    def getEntities(text):
        raise NotImplementedError("Derived classes will implement this")

class SpacyEntitiesAlgo(Algo):
    def getName():
        return SPACY_ENTITIES_ALGO

    def getEntities(text):
        doc = spacyParser(text)
        return set([ent.text for ent in doc.ents])

class SpacyNounchunksAlgo(Algo):
    def getName():
        return SPACY_NOUN_CHUNKS_ALGO

    def getEntities(text):
        doc = spacyParser(text)
        return set([ent.text for ent in doc.noun_chunks])

class GoogleEntitiesAlgo(Algo):
    def getName():
        return GOOGLE_API_ALGO

    def getEntities(text):
        result = analyze_entities(text, get_native_encoding_type())
        entitiesList = [Payload(ent) for ent in result['entities']]
        filteredEntitiesList = list(filter(lambda x: x.type=='OTHER', entitiesList))
        return set([x.name for x in filteredEntitiesList])

def getAllAlgos():
    return [SPACY_ENTITIES_ALGO, SPACY_NOUN_CHUNKS_ALGO, GOOGLE_API_ALGO]

def getAlgo(algoName):
    if algoName == SPACY_ENTITIES_ALGO:
        return SpacyEntitiesAlgo
    elif algoName == SPACY_NOUN_CHUNKS_ALGO:
        return SpacyNounchunksAlgo
    elif algoName == GOOGLE_API_ALGO:
        return GoogleEntitiesAlgo
    else:
        raise "Algo name not implemented: " + algoName

# Basic utility function to clean the text 
def clean_text(text):
    return text.strip().lower()

#Create spacy tokenizer that parses a sentence and generates tokens
#these can also be replaced by word vectors 
def spacy_tokenizer(sentence):
    tokens = spacyParser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in string.punctuation)]
    return tokens

#Custom transformer using spaCy for transforming data
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}


# Data Cleaning Utils

# Stem the words to its root entity
def stem(entity):
    return SnowballStemmer("english").stem(entity)

# Cleans text by removing unicode characters
def cleanText(entities_text):
    cleaned_text = re.sub("\s*[\d]*\s+([A-Z])",r'. \1', entities_text) #Add period after ill-formed sentences beginning with capslock.
    cleaned_text = re.sub(r'\.\.',r'.', cleaned_text) #Remove additional periods.
    cleaned_text = cleaned_text.replace('\"','') #Remove double quotes
    cleaned_text = str(cleaned_text.encode('ascii','ignore').decode('ascii')) #remove unicode characters
    return cleaned_text

# Gets the cleaned text to process from json
def getTextToProcess(entity_payload, categories_to_filter_out):
    if entity_payload.category in categories_to_filter_out:
        return ''

    entities_text = ". ".join([entity_payload.description,
                                "The category is " + entity_payload.category.lower(), 
                                "The sub category is " + entity_payload.sub_category.lower(),
                                "The title is " + entity_payload.title.lower()])

    return cleanText(entities_text)

# Cleans up detected entities by removing integers, stop words, determiners
def cleanEntities(entities):
    cleaned_entities = set(filter(is_not_null_or_integer, [ent.lower().strip('. ,;').replace('\"','') for ent in entities]))
    cleaned_entities = set([re.sub("^[a|an|the|its|his|your|it\'s|your\'s|yours|hers]+\s","", ent) for ent in cleaned_entities])
    return cleaned_entities - stopwords - set(['category', 'sub category','title']) # remove stop words and added words

# Ranker to filter entities using a pre-trained model
def L2Ranker(entities, model_file):
    if len(entities) == 0:
        return entities
    try:
        with open(model_file, 'rb') as fid:
            model = pickle.load(fid)
    except IOError:
        print("Could not read file:" + model_file)

    X = list(entities)
    Yh = model.predict(X)
    return set([X[i] for i in range(len(X)) if Yh[i] == 'pos'])

# Cleans up detected entities by removing integers, stop words, determiners, filters out non-automobile words and returns stemmed and non-stemmed versions.
def cleanAndFilterEntities(entities, model_file, filter_flag=True):
    cleaned_entities = cleanEntities(entities)
    if filter_flag:
        cleaned_entities = L2Ranker(cleaned_entities, model_file)
    stemmed_cleaned_entities = set([stem(ent) for ent in cleaned_entities])
    return (stemmed_cleaned_entities, cleaned_entities)


# Methods to process data

# Makes a key for cooccuring entity dict
def makeKey(entity1, entity2):
    return entity1 + key_delimiter + entity2

# Indicates whether one string is a subset of other
def subset(str1, str2):
    if (str1 in str2) or (str2 in str1):
        return True
    return False

# Updates count of cooccuring entities
def updateCoocccuringDict(dict, entities, sents):
    entity_count = len(entities)
    ents = list(entities)
    for i in range(0, entity_count):
        for j in range(i+1, entity_count):
            if not subset(ents[i], ents[j]):
                for sent in sents:
                    if ents[i] in sent.text and ents[j] in sent.text: # If they belong to same sentence
                        key = makeKey(ents[i], ents[j])
                        if key in dict:
                            dict[key] += 1
                        else:
                            dict[key] = 1

# Calculates count of each object
def update_dict(dict_cnt, set_of_objects):
    for obj in set_of_objects:
        if not obj in dict_cnt:
            dict_cnt[obj] = 1
        else:
            dict_cnt[obj] += 1

# Displays progress bar
def displayProgress(i,n):
    hash = ((60*i)//n)
    print("[{}{}] {}%".format('#' * hash, ' ' * (60-hash), "{0:.0f}".format((i/n)*100)), end="\r")

# Creates Json objects from json list and updates it with detected entities using different algorithms
def processData(jList, categories_to_filter_out, model_file, filter_flag=True):
    return_list=[]
    num_entries = len(jList)
    print("-------------------------------")
    print("Total entries: %d" % num_entries)
    print("-------------------------------")
    cnt=0
    all_algos = getAllAlgos()
    cooccuring_counts_dict = {}
    counts_dict = {}
    for algoName in all_algos:
        cooccuring_counts_dict[algoName] = {}
        counts_dict[algoName] = {}
    for j in jList:
        cnt+=1
        if cnt%10==0:
            displayProgress(cnt, num_entries)

        try:
            payloadJ = Payload(j)
            cleaned_entities_text = getTextToProcess(payloadJ, categories_to_filter_out)
            if not cleaned_entities_text:
                continue

            doc = spacyParser(cleaned_entities_text)
            j['id'] = cnt
            j['cleanedText'] = cleaned_entities_text
            for algoName in all_algos:
                algo = getAlgo(algoName)
                entities = algo.getEntities(cleaned_entities_text)
                (j[algoName + 'stemmed_entities'], j[algoName + 'non_stemmed_entities']) = cleanAndFilterEntities(entities, model_file, filter_flag)
                updateCoocccuringDict(cooccuring_counts_dict[algoName], j[algoName + 'stemmed_entities'], list(doc.sents))
                update_dict(counts_dict[algoName], j[algoName + 'non_stemmed_entities'])
            return_list.append(Payload(j))
        except:
            print("Encountered error: " + sys.exc_info()[0])
            print("While processing the following data: ")
            pprint(j)
            continue

    displayProgress(num_entries, num_entries)
    return return_list, cooccuring_counts_dict, counts_dict

# Relationship processing

# Reads all relations from a file
def readRelations(desc_file):
    relations = []
    try:
        with open(desc_file, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                relations.append(row)
    except IOError:
        print("Could not read file:" + desc_file)
    return relations

# Processes relations so that the main argument is separated
def processRelations(str):
    return re.sub("[(|,]"," ", str).strip().lower()

# Returns if both entities of the relation contain one of the entities in the list of entities
def containsBothEntities(relation, entities):
    subject_contains_entity = False
    object_contains_entity = False
    for entity in entities:
        if entity in processRelations(relation[2]):
            subject_contains_entity = True
        if entity in processRelations(relation[4]):
            object_contains_entity = True
    return subject_contains_entity and object_contains_entity

# Filters relations based on entities found
def filterRelations(relations_file, counts_dict):
    relations = readRelations(relations_file)
    all_algos = getAllAlgos()
    relations_dict = {}
    for algoName in all_algos:
        relations_dict[algoName] = []
        for relation in relations:
            if(containsBothEntities(relation, counts_dict[algoName].keys())):
                relations_dict[algoName].append(relation)

    return relations_dict

# Metric calculations

# Calculates precision, recall, fscore, true positives, false positives and true negatives.
def get_metrics(set_of_entities, goldenset_entities):
    true_positives=0
    false_positives=0
    false_negatives=0
    list_tp = []
    list_fp = []
    list_fn = []
    for entity in set_of_entities:
        if entity in goldenset_entities:
            true_positives+=1
            list_tp.append(entity)
        else:
            false_positives+=1
            list_fp.append(entity)

    for entity in goldenset_entities:
        if not entity in set_of_entities:
            false_negatives += 1
            list_fn.append(entity)

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    fscore = 2*precision*recall/ (precision + recall)
    return (precision, recall, fscore, list_tp, list_fp, list_fn)

# Writes debug info (true positives, false positives, true negatives) to a file for debugging purposes
def write_debug_info(entitynames, goldenset_names, tp, fp, fn, file):
    try:
        with open(file, 'w') as file_debug:
            for name in entitynames:
                output_str = name
                stemmed_name = stem(name)
                if stemmed_name in tp:
                    file_debug.write(name + ";tp\n")
                elif stemmed_name in fp:
                    file_debug.write(name + ";fp\n")
                else:
                    file_debug.write(name + ";tn\n")

            for name in goldenset_names:
                if stem(name) in fn:
                    file_debug.write(name + ";fn\n")
    except IOError:
        print("Could not read file:" + file)

# Calculates metrics and writes it to a debug file
def CalculateAndWriteMetrics(goldenset_file, counts_dict, output_directory):
    try:
        with open(goldenset_file) as goldenset_file:
            goldenset_nonstemmed_entities = set([line.rstrip('\n').lower() for line in goldenset_file])
            goldenset_entities = set([stem(ent) for ent in goldenset_nonstemmed_entities])
    except IOError:
        print("Could not read file:" + goldenset_file)

    all_algos = getAllAlgos()
    for algoName in all_algos:
        (precision, recall, fscore, list_tp, list_fp, list_fn) = get_metrics(set(stem(e) for e in counts_dict[algoName].keys()), goldenset_entities)
        print(algoName + " stats (precision, recall, fscore): " + ','.join(str(e) for e in [precision, recall, fscore]))
        write_debug_info(counts_dict[algoName].keys(), goldenset_nonstemmed_entities, list_tp, list_fp, list_fn, os.path.join(output_directory, "debug_" + algoName +".tsv"))

# Write entities to output directory

# Sort dictionary by items
def sort_dict(d):
    return [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]

# Filters out any entity with count less than 1
def return_filtered_dict(dict):
    return {k:v for k,v in dict.items() if v > 1}

# Write dictionary to a file for debugging purposes
def write_dict(d, file):
    try:
        with open(file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in d:
                writer.writerow([key, value])
    except IOError:
        print("Could not read file:" + file)

# Sorts and Writes entities of each algo to output directory
def SortAndWriteEntities(counts_dict, output_directory, coocur_flag=False):
    all_algos = getAllAlgos()
    for algoName in all_algos:
        entities_cnt = counts_dict[algoName]
        prefix = "count_"
        if coocur_flag:
            entities_cnt = return_filtered_dict(entities_cnt)
            prefix = "cooccur_"
        sorted_dict = sort_dict(entities_cnt)
        write_dict(sorted_dict, os.path.join(output_directory, prefix + algoName +".tsv"))