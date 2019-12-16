
from AISD_addALL_TrainTest_Pretrain_df import config
import numpy as np
from tensorflow.contrib import learn
from AISD_addALL_TrainTest_Pretrain_df.negativeSample import InitNegTable
import random
from sklearn.preprocessing import LabelBinarizer


class dataSet:
	def __init__(self,text_path,graph_path,domain_path,trainid_path,trainClaId_path_hotel,trainClaId_path_res,testid_hotel,testid_res,pretrain_emb_path,trainmode='train'):

		text_file,graph_file=self.load(text_path,graph_path)

		self.train_id,self.test_id,self.trainCla_id = self.loadTrainTest(trainid_path,trainClaId_path_hotel,trainClaId_path_res,testid_hotel,testid_res)
		print(len(self.trainCla_id),len(self.test_id))
		self.train_edges,self.test_edges=self.load_trainTestEdges(graph_file)
		self.trainCla_edges = self.loadTrainClaEdge()

		if trainmode == 'trainAddTest':
			self.train_edges.extend(self.test_edges)

		self.train_edges, self.test_edges = self.encodeGraph(self.train_edges, self.test_edges)

		self.text, self.num_vocab, self.num_nodes, self.vocab_map=self.load_text(text_file)
		self.emb_matrix = self.loadEmbMatrix(pretrain_emb_path, self.vocab_map, self.num_vocab)
		self.domain = self.load_domain(domain_path)

		self.text_neg_table, self.num_texts, self.user_neg_table, self.num_users, self.item_neg_table, self.num_items \
			, self.loc_neg_table, self.num_locs, self.rating_neg_table, self.num_ratings, self.date_neg_table, self.num_dates \
			, self.price_neg_table, self.num_prices, self.wifi_neg_table, self.num_wifis, self.acc_neg_table, self.num_accs \
			, self.web_neg_table, self.num_webs, self.phone_neg_table, self.num_phones = InitNegTable(self.train_edges)
	def encodeGraph(self,train_edges,test_edges):
		userlist = []
		itemlist = []
		loclist = []
		ratinglist = []
		datelist = []
		priceRangelist = []
		wifilist = []
		acclist = []
		weblist = []
		phonelist = []
		for line in train_edges:
			splits = line
			user = str(splits[1]).strip()
			item = str(splits[2]).strip()
			user_loc = str(splits[3]).strip()
			item_loc = str(splits[4]).strip()
			review_rating = str(splits[5]).strip()
			item_rating = str(splits[6].strip())

			user_joindate = str(splits[7]).strip()
			review_date = str(splits[8]).strip()
			price_range = str(splits[9]).strip()
			wifi = str(splits[10]).strip()
			acc = str(splits[11]).strip()
			web = str(splits[12]).strip()
			phone = str(splits[13]).strip()

			userlist.append(user)
			itemlist.append(item)
			loclist.append(user_loc)
			loclist.append(item_loc)
			ratinglist.append(review_rating)
			ratinglist.append(item_rating)
			datelist.append(user_joindate)
			datelist.append(review_date)
			priceRangelist.append(price_range)
			wifilist.append(wifi)
			acclist.append(acc)
			weblist.append(web)
			phonelist.append(phone)
		user_dict = self.getDict(userlist)
		item_dict = self.getDict(itemlist)
		loc_dict = self.getDict(loclist)
		rating_dict = self.getDict(ratinglist)
		date_dict = self.getDict(datelist)
		priceRange_dict = self.getDict(priceRangelist)
		wifi_dict = self.getDict(wifilist)
		acc_dict = self.getDict(acclist)
		web_dict = self.getDict(weblist)
		phone_dict = self.getDict(phonelist)

		trainedges_encode = []
		for trainedge in train_edges:
			splits = trainedge
			reviewindex = str(splits[0]).strip()
			reviewerID = user_dict[str(splits[1]).strip()] if str(splits[1]).strip() in user_dict else user_dict['UNKNOWN']
			hotelID = item_dict[str(splits[2]).strip()] if str(splits[2]).strip() in item_dict else item_dict['UNKNOWN']
			user_loc = loc_dict[str(splits[3]).strip()] if str(splits[3]).strip() in loc_dict else loc_dict['UNKNOWN']
			item_loc = loc_dict[str(splits[4]).strip()]	if str(splits[4]).strip() in loc_dict else loc_dict['UNKNOWN']
			rating = rating_dict[str(splits[5]).strip()] if str(splits[5]).strip() in rating_dict else rating_dict['UNKNOWN']
			item_rating = rating_dict[str(splits[6]).strip()] if str(splits[6]).strip() in rating_dict else rating_dict['UNKNOWN']

			user_joindate = date_dict[str(splits[7]).strip()] if str(splits[7]).strip() in date_dict else date_dict['UNKNOWN']
			review_date = date_dict[str(splits[8]).strip()] if str(splits[8]).strip() in date_dict else date_dict['UNKNOWN']
			price_range = priceRange_dict[str(splits[9]).strip()] if str(splits[9]).strip() in  priceRange_dict else priceRange_dict['UNKNOWN']
			wifi = wifi_dict[str(splits[10]).strip()] if str(splits[10]).strip() in wifi_dict else wifi_dict['UNKNOWN']
			acc = acc_dict[str(splits[11]).strip()] if str(splits[11]).strip() in acc_dict else acc_dict['UNKNOWN']
			web = web_dict[str(splits[12]).strip()] if str(splits[12]).strip() in web_dict else web_dict['UNKNOWN']
			phone = phone_dict[str(splits[13]).strip()] if str(splits[13]).strip() in phone_dict else phone_dict['UNKNOWN']


			train_encode = [reviewindex,reviewerID,hotelID,user_loc,item_loc,rating,item_rating,user_joindate,review_date,price_range,wifi,acc,web,phone]
			trainedges_encode.append(train_encode)


		testedges_encode = []
		for testedge in test_edges:
			splits = testedge
			reviewindex = str(splits[0]).strip()
			reviewerID = user_dict[str(splits[1]).strip()] if str(splits[1]).strip() in user_dict else user_dict['UNKNOWN']
			hotelID = item_dict[str(splits[2]).strip()] if str(splits[2]).strip() in item_dict else item_dict['UNKNOWN']
			user_loc = loc_dict[str(splits[3]).strip()] if  str(splits[3]).strip() in loc_dict else loc_dict['UNKNOWN']
			item_loc = loc_dict[str(splits[4]).strip()]	if str(splits[4]).strip() in loc_dict else loc_dict['UNKNOWN']
			rating = rating_dict[str(splits[5]).strip()] if str(splits[5]).strip() in rating_dict else rating_dict['UNKNOWN']
			item_rating = rating_dict[str(splits[6]).strip()] if str(splits[6]).strip() in rating_dict else rating_dict['UNKNOWN']

			user_joindate = date_dict[str(splits[7]).strip()] if str(splits[7]).strip() in date_dict else date_dict['UNKNOWN']
			review_date = date_dict[str(splits[8]).strip()] if str(splits[8]).strip() in date_dict else date_dict['UNKNOWN']
			price_range = priceRange_dict[str(splits[9]).strip()] if str(splits[9]).strip() in priceRange_dict else priceRange_dict['UNKNOWN']
			wifi = wifi_dict[str(splits[10]).strip()] if str(splits[10]).strip() in wifi_dict else wifi_dict['UNKNOWN']
			acc = acc_dict[str(splits[11]).strip()] if str(splits[11]).strip() in acc_dict else acc_dict['UNKNOWN']
			web = web_dict[str(splits[12]).strip()] if str(splits[12]).strip() in web_dict else web_dict['UNKNOWN']
			phone = phone_dict[str(splits[13]).strip()] if str(splits[13]).strip() in phone_dict else phone_dict['UNKNOWN']

			test_encode = [reviewindex,reviewerID,hotelID,user_loc,item_loc,rating,item_rating,user_joindate,review_date,price_range,wifi,acc,web,phone]
			testedges_encode.append(test_encode)


		return trainedges_encode,testedges_encode

	def loadTrainClaEdge(self):
		trainCla_edges = []
		for edge in self.train_edges:
			if int(edge[0]) in self.trainCla_id:
				trainCla_edges.append(edge)
		return trainCla_edges
	def getDict(self,inputlist):
		list_clean = list(set(inputlist))
		index = 0
		dict = {}
		for id in list_clean:
			dict[id] = index
			index = index + 1
		dict['UNKNOWN'] = index
		return dict
	def load(self,text_path,graph_path):
		text_file=open(text_path,'rb',errors=None).readlines()
		graph_file=open(graph_path,'rb',errors=None).readlines()
		return text_file,graph_file

	def load_domain(self,domain_file):
		f = open(domain_file,"rb")
		domainlabel = []
		for line in f.readlines():
			domainlabel.append(line.strip())

		lb = LabelBinarizer()
		domainlabels = lb.fit_transform(domainlabel)

		return domainlabels
	def loadEmbMatrix(self,pretrain_emb_path,vocab_map,num_vocab):
		print('Indexing word vectors.')
		embeddings_index = {}
		f = open(pretrain_emb_path,'rb')
		count = 0
		for line in f:
			count = count + 1
			if (count < 2):
				continue
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		print('Found %s word vectors.' % len(embeddings_index))

		nb_words = num_vocab
		embedding_matrix = np.zeros((nb_words + 1, config.embed_size))
		for word, i in vocab_map.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		return  embedding_matrix

	def loadTrainTest(self,trainid_path,trainClaId_path_hotel,trainClaId_path_res,testid_hotel,testid_res):
		trainids = open(trainid_path, 'rb').readlines()
		train_id = [int(id.strip()) for id in trainids]
		testids_hotel = open(testid_hotel, 'rb').readlines()
		test_id_hotel = [int(id.strip()) for id in testids_hotel]
		testids_res = open(testid_res, 'rb').readlines()
		test_id_res = [int(id.strip()) for id in testids_res]
		test_id = test_id_hotel
		test_id.extend(test_id_res)


		trainClaids_hotel = open(trainClaId_path_hotel, 'rb').readlines()
		trainCla_id_hotel = [int(id.strip()) for id in trainClaids_hotel]

		trainClaids_res = open(trainClaId_path_res, 'rb').readlines()
		trainCla_id_res = [int(id.strip()) for id in trainClaids_res]
		trainCla_id=trainCla_id_hotel
		trainCla_id.extend(trainCla_id_res)
		return train_id,test_id,trainCla_id



	def load_trainTestEdges(self,graph_file):

		train_edges=[]
		test_edges=[]
		for i in graph_file:
			edgesplit = i.decode().strip().split('\t')
			if int(edgesplit[0]) in self.train_id:
				train_edges.append(edgesplit)
			elif int(edgesplit[0]) in self.test_id:
				test_edges.append(edgesplit)
			# else:
			# 	print("Not in : "+i.strip())

		return train_edges,test_edges

	def load_text(self,text_file):
		vocab=learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
		for i in range(len(text_file)):
			#print(i)
			text_file[i]=text_file[i].decode()
		text=np.array(list(vocab.fit_transform(text_file)))
		num_vocab=len(vocab.vocabulary_)
		num_nodes=len(text)

		return text,num_vocab,num_nodes,vocab.vocabulary_._mapping

	def negative_sample(self,edges):
		text,user,item,user_loc,item_loc,review_rating,item_rating,user_jdate,review_date,item_price,item_wifi,item_acc,item_web,item_phone=zip(*edges)
		sample_edges=[]
		func=lambda table: table[random.randint(0, config.neg_table_size - 1)]
		for i in range(len(edges)):
			text_neg = func(self.text_neg_table)
			while text[i] == text_neg:
				text_neg = func(self.text_neg_table)

			user_neg = func(self.user_neg_table)
			while user[i] == user_neg:
				user_neg = func(self.user_neg_table)

			item_neg = func(self.item_neg_table)
			while item[i] == item_neg:
				item_neg = func(self.item_neg_table)

			user_loc_neg = func(self.loc_neg_table)
			while user_loc[i] == user_loc_neg:
				user_loc_neg = func(self.loc_neg_table)

			item_loc_neg = func(self.loc_neg_table)
			while item_loc[i] == item_loc_neg:
				item_loc_neg = func(self.loc_neg_table)

			review_rating_neg = func(self.rating_neg_table)
			while review_rating[i] == review_rating_neg:
				review_rating_neg = func(self.rating_neg_table)

			item_rating_neg = func(self.rating_neg_table)
			while item_rating[i] == item_rating_neg:
				item_rating_neg = func(self.rating_neg_table)

			user_jdate_neg = func(self.date_neg_table)
			while user_jdate[i] == user_jdate_neg:
				user_jdate_neg = func(self.date_neg_table)

			review_date_neg = func(self.date_neg_table)
			while review_date[i] == review_date_neg:
				review_date_neg = func(self.date_neg_table)

			item_price_neg = func(self.price_neg_table)
			while item_price[i] == item_price_neg:
				item_price_neg = func(self.price_neg_table)

			item_wifi_neg = func(self.wifi_neg_table)
			while item_wifi[i] == item_wifi_neg:
				item_wifi_neg = func(self.wifi_neg_table)

			item_acc_neg = func(self.acc_neg_table)
			while item_acc[i] == item_acc_neg:
				item_acc_neg = func(self.acc_neg_table)

			item_web_neg = func(self.web_neg_table)
			while item_web[i] == item_web_neg:
				item_web_neg = func(self.web_neg_table)

			item_phone_neg = func(self.phone_neg_table)
			while item_phone[i] == item_phone_neg:
				item_phone_neg = func(self.phone_neg_table)


			sample_edges.append([text[i],user[i],item[i],user_loc[i],item_loc[i],review_rating[i],item_rating[i],user_jdate[i],review_date[i],item_price[i],item_wifi[i],item_acc[i],item_web[i],item_phone[i]\
								,text_neg,user_neg,item_neg,user_loc_neg,item_loc_neg,review_rating_neg,item_rating_neg,user_jdate_neg,review_date_neg,item_price_neg,item_wifi_neg,item_acc_neg,item_web_neg,item_phone_neg])
		return sample_edges

	def generate_batches(self,mode=None,trainmode='train'):
		num_batch = int(len(self.train_edges) / config.batch_size)
		edges = self.train_edges

		if mode=='add':
			if trainmode == 'train':
				edges.extend(self.test_edges)
				size = len(edges)
				num_batch = int( size / config.batch_size)
				num_batch+=1
				edges.extend(edges[:(config.batch_size - size % config.batch_size)])
			elif trainmode == 'trainAddTest':
				size = len(edges)
				num_batch += 1
				edges.extend(edges[:(config.batch_size - size % config.batch_size)])
			else:
				print('wrong trainmode')
		if mode != 'add':
			np.random.shuffle(edges)
		sample_edges= edges[:int(num_batch * config.batch_size)]
		sample_edges=self.negative_sample(sample_edges)
		print(len(sample_edges))


		batches=[]
		for i in range(num_batch):
			batches.append(sample_edges[i * config.batch_size:(i + 1) * config.batch_size])
		# print sample_edges[0]
		return batches

