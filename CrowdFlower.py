import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import sklearn
import ast


def dprint(desc, item):
	print ("-" * 20 + str(desc) + "-" * 20)
	print (item)
	print ("-" * 40)

def iprint(text):
	print ("INFO: " + str(text))

def save_checkpoint(filename, dataframe):
	dataframe.to_csv(filename + ".csv")

def get_checkpoint(filename):
	try:
		dataframe = pd.read_csv(filename + ".csv")
		iprint("Checkpoint found: " + str(filename))
		return dataframe
	except IOError:
		iprint("No checkpoint found.")
	return None

'''
s = sentiment
w = when
k = kind

s1,"I can't tell"
s2,"Negative"
s3,"Neutral / author is just sharing information"
s4,"Positive"
s5,"Tweet not related to weather condition"

w1,"current (same day) weather"
w2,"future (forecast)"
w3,"I can't tell"
w4,"past weather"

k1,"clouds"
k2,"cold"
k3,"dry"
k4,"hot"
k5,"humid"
k6,"hurricane"
k7,"I can't tell"
k8,"ice"
k9,"other"
k10,"rain"
k11,"snow"
k12,"storms"
k13,"sun"
k14,"tornado"
k15,"wind"
'''
label_words = [
	"clouds",
	"cold",
	"dry",
	"hot",
	"humid",
	"hurricane",
	#"nothing", #"I can't tell",
	"ice",
	"other",
	"rain",
	"snow",
	"storms",
	"sun",
	"tornado",
	"wind",
]

label_wups = []
for word in label_words:
	label_wups.append(wn.synsets(word)[0])
print label_wups

lemmatizer = WordNetLemmatizer()

iprint("Reading dataset")
pd.set_option('display.max_columns', None)
train_set = pd.read_csv("CrowdFlowerData/train.csv")
dprint ("train head", train_set.head())

train_set.loc[:,"tweet"] = train_set.loc[:, "tweet"].fillna("")
dprint ("train head", train_set.head())

# we will then remove the @mentions
train_set["tweet"] = train_set["tweet"].str.replace("@mention:", "")
dprint ("train head", train_set.head())


has_rt = train_set["tweet"].str.contains("RT")
train_set["has_rt"] = has_rt
dprint ("train head", train_set.head())

has_mention = train_set["tweet"].str.contains("@mention")
train_set["has_mention"] = has_mention
dprint ("train head", train_set.head())

checkpoint_data = get_checkpoint("tokenized_tweets")
if (checkpoint_data is None):
	iprint("tokenizing tweets")
	train_set["tokenized_tweets"] = train_set["tweet"].apply(nltk.word_tokenize)
	dprint ("train head", train_set.head())
	save_checkpoint("tokenized_tweets", train_set)
else:
	train_set = checkpoint_data
	dprint ("train head", train_set.head())

cache = set(stopwords.words('english'))
cache2 = set(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#', '%', '!', '$', '^', '&', '*', '...'])

def remover(words_list):
	if (type(words_list) == str):
		words_list = ast.literal_eval(words_list)
	new_list = []
	for word in words_list:
		if str(word).lower() not in cache and word not in cache2:
			new_list.append(word.lower())
	return new_list

checkpoint_data = get_checkpoint("removed_stopwords")
if(checkpoint_data is None):
	iprint("removing stop words")
	cache = stopwords.words('english')
	train_set["removed_stopwords"] = train_set["tokenized_tweets"].apply(remover)
	dprint ("train head", train_set.head())
	save_checkpoint("removed_stopwords", train_set)
else:
	train_set = checkpoint_data
	dprint ("train head", train_set.head())

def lemma_wrapper(words_list):
	if (type(words_list) == str):
		words_list = ast.literal_eval(words_list)
	return [lemmatizer.lemmatize(word) for word in words_list]

checkpoint_data = get_checkpoint("lemmatized_words")
if(checkpoint_data is None):
	iprint("lemmatizing words")
	train_set["lemmatized_words"] = train_set["removed_stopwords"].apply(lemma_wrapper)
	dprint("train head", train_set.head())
	save_checkpoint("lemmatized_words", train_set)
else:
	train_set = checkpoint_data
	dprint ("train head", train_set.head())

def pos_tag_wrapper(words):
	return nltk.pos_tag(words)

checkpoint_data = get_checkpoint("pos_tagged")
if(checkpoint_data is None):
	iprint("Part of Speech Tagging")
	train_set["pos_tagged"] = train_set["removed_stopwords"].apply(pos_tag_wrapper)
	dprint("train head", train_set.head())
	save_checkpoint("pos_tagged", train_set)
else:
	train_set = checkpoint_data
	dprint("train head", train_set.head())


def numerize_words(words):

	word_vec = dict.fromkeys(label_wups, 0)
	words = ast.literal_eval(words)

	for word in words:
		try:
			target_word = wn.synsets(word)[0]
		except IndexError:
			target_word = wn.synsets("nothing")[0]
			continue

		for label in label_wups:
			cross_value = label.wup_similarity(target_word)
			if (cross_value == None):
				cross_value = 0
			word_vec[label] += cross_value
	best_val = 0
	best_label = ""
	for key, value in word_vec.iteritems():
		
		print key
		print value

		if (value > best_val):
			best_val = value
			best_label = key

	print ("Best label is: " + str(best_label))
	return best_label

checkpoint_data = get_checkpoint("numerized_words")
if(checkpoint_data is None):
	iprint("Numerizing words with wup similarity")
	train_set["numerized_words"] = train_set["lemmatized_words"].apply(numerize_words)
	train_set["numerized_words2"] = train_set["removed_stopwords"].apply(numerized_words)
	dprint("train head", train_set.head())
	save_checkpoint("numerized_words", train_set)
else:
	train_set = checkpoint_data
	dprint("train head", train_set.head())



'''
Text Process

Segmentation

PoS Tagging

Filtering

Lemmatization

Clustering

TF-IDF

SVD

'''



