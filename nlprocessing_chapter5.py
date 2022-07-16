import spacy

nlp = spacy.load("en_core_web_sm")
text = "West chestertenfieldville was referenced in Mr. Deeds."
doc = nlp(text)
#entity ruler. when to implement it. add custom features into spacy. rules based approach and a machine based approach. 
#rule based approach is either a list a set of rules ina list of unknown lists. regex code and linguistic features. the rules need to write out. entity recognition. extract dates from text. 
#i could have like January 2022. different ways we can do this. regex expression for to capture all of those. 
#rule-based approach for are something like names . like poeple. geneate entity ruler a list of possibile first names and last names prefices doctor, ms, mr, senior etc. 
#difficult to write thee quanity of first and last names in the world is massive.
#for this reason we work on machine learning component. the rules based approach is a good and practitioner is machine learning approaches. maybe this task is not appropriate so that we should have a high degree of confidence. If we are okay with the true or false positives getting to know our documentation. a list of series feratures to transform them into token . lets go ahead and do this right now. 
#if we have got the reference congrats.
#in this work, we would like to extract the text we are oing to be working with. 
#we want our model or our pipeline. 
for ent in doc.ents:
 print(ent.text, ent.label_)

#make a ruler to correct this problem. 
ruler = nlp.add_pipe("entity_ruler")
nlp.analyze_pipes()

#a list of dictionary
patterns = [{"label": "GPE" ,"pattern" :"West Chestertenfieldville" }]
ruler.add_patterns(patterns)
doc2 = nlp(text)
for ent in doc2.ents:
 print(ent.text, ent.label_)


nlp2 = spacy.load("en_core_web_sm")
ruler = nlp2.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

doc = nlp2(text)
for ent in doc.ents:
 print(ent.text, ent.label_)


nlp3 = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
 {"label": "GPE", "pattern": "West Chestertenfieldville"},
 {"label": "FILM", "pattern": "Mr. Deeds"}


doc = nlp3(text)
for ent in doc.ents:
 print(ent.text, ent.label_)


#toponym resolution if we look at this word this GPE is referred to Paris to resolve problems
#like this. in word embeddings when dr deeds means it is a film and when does it mean it is a
#person lets move on to the example: spacy can do a lot of things check and see mr deeds
#the answer will come up to in a while. what I can do in spacy take the linguistic sequences
#we start talking about matcher more robust things I can do with spacy or other fancier things with approacher. 
#The most important areas in this video
#We are going to be doing a lot with spacy.
#vocab NLP model. lexemp. dot dot ents. we want to user the entity ruler. The entities that we are copying right now. 
#Holocaust data we are trying to data games along side other entities. When I use the matter it is not necessarily entity type that will help me extract information. 
#We are kinda use the matter it is implemented into entity rule.
#when we want to use the rule the lemma of the word identify the word over regex. 
#complicated pattern we need to extract that pattern does not depend on a specific part of a speech is when to user REGEX. 

