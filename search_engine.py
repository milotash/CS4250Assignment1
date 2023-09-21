#-------------------------------------------------------------------------
# AUTHOR: Ismael Garcia
# FILENAME: search_engine.py
# SPECIFICATION: This program uses a CSV file to get the documents needed. We use those documents to calculate the precision and recall percentages.
# FOR: CS 4250- Assignment #1
# TIME SPENT: Too long. Had to figure out Python.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard arrays

#importing some Python libraries
import csv
import math

q = ['cat and dogs']
retrievalScore = 0.1

documents = []
labels = []

#reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])
            labels.append(row[1])

#Printing to show contents of original documents.
print("Contents of documents: ", documents)
print()

#Printing to show contents of the documents after tokenizing them.
tokenDoc = [string.split() for string in documents]
print("Tokenized documents: ", tokenDoc)
print()

#Conduct stopword removal.
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}

#This is removing any terms that are in stopWords from the documents.
newTokenDoc = []
for doc in tokenDoc:
    tempDoc = []
    for token in doc:
        if token not in stopWords:
            tempDoc.append(token)
    newTokenDoc.append(tempDoc)

#Printing to show contents of the documents after applying stop word removal.
print("After removing stopWords: ", newTokenDoc)
print()

#Conduct stemming.
steeming = {
  "cats": "cat",
  "dogs": "dog",
  "loves": "love",
}

#This part is applying stemming to the contents of the documents.
for i in range(len(newTokenDoc)):
    for j in range(len(newTokenDoc[i])):
        if newTokenDoc[i][j] in steeming:
            newTokenDoc[i][j] = steeming[newTokenDoc[i][j]]

#Printing to show contents of the documents after applying stemming.
print("After stemming: ", newTokenDoc)
print()

#This is just to show the documents more clearly and printing them.
doc1 = newTokenDoc[0]
doc2 = newTokenDoc[1]
doc3 = newTokenDoc[2]
print("doc1: ", doc1)
print("doc2: ", doc2)
print("doc3: ", doc3)
print()

#Identify the index terms.
terms = []

#Creates unique index terms.
for doc in newTokenDoc:
    for term in doc:
        if term not in terms:
            terms.append(term)

#Printing to show the index terms.
print("Index terms: ", terms)
print()

#Initionizing tf to have the same number of terms and same number of documents as the newTokenDoc.
tf = [ [0] * len(terms) for term in range(len(newTokenDoc)) ]

#Calculates tf (term frequency) for each term in each document.
for i, doc in enumerate(newTokenDoc):
    total_terms_in_doc = len(doc)
    for j, term in enumerate(terms):
        term_count = doc.count(term)
        tf[i][j] = term_count / total_terms_in_doc

#Printing to show contents of tf values of the documents.
print("tf values of docs: ", tf)
print()

#Calculate idf (inverse document frequency) for each term in each document.
idf = []
for term in terms:
    term_count = sum(1 for doc in newTokenDoc if term in doc)
    idf_value = math.log( (len(newTokenDoc) / term_count), 10 )
    idf.append(idf_value)

#Printing to show the idf values of the documents.
print("idf values: ", idf)
print()

#Build the tf-idf term weights matrix.
docMatrix = []
for row in tf:
    docRow = [term * idf_value for term, idf_value in zip(row, idf)]
    docMatrix.append(docRow)

#Printing to show tf-idf values.
print("docMatrix: ", docMatrix)
print()

#This is just to print the tf-idf matrix visually.
print("Visual of tf-idf Matrix")
print(f"{' ':<10}", end="")
for term in terms:
    print(f"{term:<15}", end="")
print()
for i, doc_vector in enumerate(docMatrix, 1):
    print(f"doc{i:<5}", end="")
    for value in doc_vector:
        print(f"{value:<15.6f}", end="")
    print()
print()

#Tokenizing, stopping, and stemming, and printing the query.
tokenQuery = q[0].split()
tokenQuery = [token for token in tokenQuery if token.lower() not in stopWords]
stemQuery = [steeming.get(token.lower(), token) for token in tokenQuery]
print("Query after tokenizing, stopping, and stemming: ", stemQuery)
print()

#Applying binary weights to the query: 1 if it has the term and 0 if not. Then printing it.
queryWeights = []
for term in terms:
    if term in stemQuery:
        queryWeights.append(1)
    else:
        queryWeights.append(0)
print("query weights: ", queryWeights)
print()

#Calculate the document scores (ranking) using document weigths (tf-idf) calculated before and query weights (binary - have or not the term).
docScores = []
#Dot-product of query weights with tf-idf of the documents.
for doc_vector in docMatrix:
    score = sum(query * weight for query, weight in zip(queryWeights, doc_vector))
    docScores.append(score)

#Printing to show the document scores.
print("Document scores: ", docScores)
print()

#Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
hitDocs = []
noiseDocs = []
missedDocs = []
rejectedDocs = []
for i, (label, score) in enumerate(zip(labels, docScores), 1):
    if label == ' R' and score >= retrievalScore:
        hitDocs.append(i)
    elif label == ' I' and score >= retrievalScore:
        noiseDocs.append(i)
    elif label == ' R' and score < retrievalScore:
        missedDocs.append(i)
    elif label == ' I' and score < retrievalScore:
        rejectedDocs.append(i)

#Printing to show the document number of the hit, noise, missed, and rejected documents.
print("Hit documents: ", hitDocs)
print("Noise documents: ", noiseDocs)
print("Missed documents: ", missedDocs)
print("Rejected documents: ", rejectedDocs)
print()

#Calculating precision and recall.
precision = ( len(hitDocs) / ( len(hitDocs) + len(noiseDocs) ) ) * 100
recall = ( len(hitDocs) / ( len(hitDocs) + len(missedDocs) ) ) * 100

#Printing to show the precision and recall percentages.
print("Precision: ", precision, '%')
print("Recall: ", recall, '%')
