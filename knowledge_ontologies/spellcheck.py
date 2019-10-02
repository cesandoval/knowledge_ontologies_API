from textblob import Word

def spellcheck(word):
	word_object=Word(word)
	return word_object.correct()
	# possibilities=word_object.spellcheck()

	# if possibilities[0][1]>=.6:
	# 	return possibilities[0][0]
	# else:
	# 	print('Inconclusive inputted text,'+word+ ' could be a variety of words 2333333')
	# 	print (possibilities)
	# 	return word
# print(spellcheck('rn'))
