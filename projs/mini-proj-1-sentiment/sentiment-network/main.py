from sentiment_network import SentimentNetwork

if __name__ == '__main__':
	# open reviews and labels
	with open('reviews.txt', 'r') as review_file, \
		 open('labels.txt', 'r')  as label_file:
		reviews = review_file.read().split('\n')
		labels = label_file.read().upper().split('\n')

	# preprocess the data
	# reviews:
	# 	gather the vocabs of each word in reviews
	# labels:
	#	gather the vocabs in labels
	#	make labels to be all cap
	
	my_nn = SentimentNetwork()
	# my_nn.preprocess_data(reviews,labels,min_count=50,polarity_cutoff=0.05)
	# my_nn.init_network(num_input_nodes=len(my_nn.word_vocab),
	# 				   num_hidden_nodes=10,
	# 				   num_output_nodes=1,
	# 				   learn_rate=0.1)	
	
	# my_nn.test(reviews[-1000:],labels[-1000:])
	# my_nn.train(reviews[:-1000],labels[:-1000])
	# my_nn.test(reviews[-1000:],labels[-1000:])

	my_nn.preprocess_data(reviews,labels,min_count=50,polarity_cutoff=0.05)
	my_nn.init_network(num_input_nodes=len(my_nn.word_vocab),
					   num_hidden_nodes=10,
					   num_output_nodes=1,
					   learn_rate=0.1)	
	my_nn.test(reviews[-1000:],labels[-1000:])
	my_nn.train(reviews[:-1000],labels[:-1000])
	my_nn.test(reviews[-1000:],labels[-1000:])

	my_nn.preprocess_data(reviews,labels,min_count=50,polarity_cutoff=0.8)
	my_nn.set_learn_rate(learn_rate=0.1)
	my_nn.test(reviews[-1000:],labels[-1000:])
	my_nn.train(reviews[:-1000],labels[:-1000])
	my_nn.test(reviews[-1000:],labels[-1000:])