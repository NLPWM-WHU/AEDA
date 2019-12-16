# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import os
from AISD_addALL_TrainTest_Pretrain_df import config, AIE_add
from AISD_addALL_TrainTest_Pretrain_df.DataSet import dataSet
from datetime import datetime
np.random.seed(6666)
#load data
basedir = 'F:/txy\领域适应\dataInput/'
graph_path=basedir+'graph.txt'
text_path=basedir+'data2.txt'

trainid_path = basedir + 'ColdStart_Update_twoDomain/trainEmb.txt'
trainClaId_path_hotel = basedir + 'ColdStart_Update_hotel/train.txt'
trainClaId_path_res = basedir + 'ColdStart_Update_restaurant/train.txt'
testid_hotel = basedir + 'ColdStart_Update_hotel/test.txt'
testid_res = basedir + 'ColdStart_Update_restaurant/test.txt'
pretrain_emb_path = basedir+"review_shuffle_w2v_c1w8-i20h0n5s100.txt"
domain_path=basedir+'domainlabel.txt'
trainmode = 'train'


data=dataSet(text_path,graph_path,domain_path,trainid_path,trainClaId_path_hotel,trainClaId_path_res,testid_hotel,testid_res,pretrain_emb_path,trainmode=trainmode)

# start session

with tf.Graph().as_default():
	sess=tf.Session()
	with sess.as_default():
		model= AIE_add.Model(data.num_vocab, data.num_users, data.num_items, data.num_locs, data.num_ratings, data.num_dates, data.num_prices,data.num_wifis,data.num_accs,data.num_webs,data.num_phones,data.emb_matrix)
		opt=tf.train.AdamOptimizer(config.lr)

		train_op_user = opt.minimize(model.userloss)
		train_op_item = opt.minimize(model.itemloss)
		train_op_review = opt.minimize(model.reviewloss)
		train_op_combined = opt.minimize(model.combineloss)

		train_op_domain = opt.minimize(model.domainloss,var_list= [model.Wdomain,model.bdomain])
		train_op_domainfusion = opt.minimize(model.domainfusionloss, var_list=[model.text_embed,model.user_embed,model.item_embed,model.loc_embed,model.rating_embed,model.date_embed,model.price_embed,model.wifi_embed,model.acc_embed,model.web_embed,model.phone_embed,model.W2,model.b])
		sess.run(tf.global_variables_initializer())

		#training
		print('start training.......')

		for epoch in range(config.num_epoch):
			start = datetime.now()
			userloss_epoch = 0
			itemloss_epoch = 0
			reviewloss_epoch = 0
			combinedloss_epoch = 0
			domainloss_epoch = 0
			domainfusionloss_epoch = 0
			batches=data.generate_batches()
			h1=0
			num_batch=len(batches)
			for i in range(num_batch):
				batch=batches[i]

				textnode, user, item, user_loc, item_loc, review_rating, item_rating, user_jdate, review_date, item_price, item_wifi, item_acc, item_web, item_phone \
					, text_neg_node, user_neg, item_neg, user_loc_neg, item_loc_neg, review_rating_neg, item_rating_neg, user_jdate_neg, review_date_neg, item_price_neg, item_wifi_neg, item_acc_neg, item_web_neg, item_phone_neg = zip(
					*batch)
				textnode, user, item, user_loc, item_loc, review_rating, item_rating, user_jdate, review_date, item_price, item_wifi, item_acc, item_web, item_phone \
					, text_neg_node, user_neg, item_neg, user_loc_neg, item_loc_neg, review_rating_neg, item_rating_neg, user_jdate_neg, review_date_neg, item_price_neg, item_wifi_neg, item_acc_neg, item_web_neg, item_phone_neg = \
					np.array(textnode), np.array(user), np.array(item), np.array(user_loc), np.array(
						item_loc), np.array(review_rating), np.array(item_rating), np.array(user_jdate), np.array(
						review_date), np.array(item_price), np.array(item_wifi), np.array(item_acc), np.array(
						item_web), np.array(item_phone) \
						, np.array(text_neg_node), np.array(user_neg), np.array(item_neg), np.array(
						user_loc_neg), np.array(item_loc_neg), np.array(review_rating_neg), np.array(
						item_rating_neg), np.array(user_jdate_neg), np.array(review_date_neg), np.array(
						item_price_neg), np.array(item_wifi_neg), np.array(item_acc_neg), np.array(
						item_web_neg), np.array(item_phone_neg)
				textnode = list(map(int,textnode))
				text_neg_node = list(map(int,text_neg_node))
				text,text_neg=data.text[textnode],data.text[text_neg_node]
				domain = data.domain[textnode]
				uniform = []
				for j in range(len(batch)):
					uniform.append([0.5,0.5])
				uniform = np.array(uniform)

				feed_dict={
					model.Text : text,
					model.User : user,
					model.Item : item,
					model.User_loc : user_loc,
					model.Item_loc : item_loc,
					model.Review_rating : review_rating,
					model.Item_rating : item_rating,
					model.User_jdate : user_jdate,
					model.Review_date: review_date,
					model.Item_price:item_price,
					model.Item_wifi: item_wifi,
					model.Item_web:item_web,
					model.Item_phone:item_phone,
					model.Item_acc: item_acc,
					model.Text_neg : text_neg,
					model.User_neg : user_neg,
					model.Item_neg : item_neg,
					model.User_loc_neg : user_loc_neg,
					model.Item_loc_neg : item_loc_neg,
					model.Review_rating_neg : review_rating_neg,
					model.Item_rating_neg: item_rating_neg,
					model.User_jdate_neg : user_jdate_neg,
					model.Review_date_neg: review_date_neg,
					model.Item_price_neg:item_price_neg,
					model.Item_wifi_neg: item_wifi_neg,
					model.Item_acc_neg: item_acc_neg,
					model.Item_web_neg: item_web_neg,
					model.Item_phone_neg: item_phone_neg,
					model.Domain: domain,
					model.dropout_keep_prob: 1.0,
					model.Uniform : uniform
				}

				# run the graph

				# userloss_batch =0
				# itemloss_batch = 0
				# reviewloss_batch = 0
				_,userloss_batch = sess.run([train_op_user,model.userloss],feed_dict=feed_dict)
				_,itemloss_batch = sess.run([train_op_item, model.itemloss], feed_dict=feed_dict)
				_,reviewloss_batch = sess.run([train_op_review, model.reviewloss], feed_dict=feed_dict)
				#combinedloss_batch = 0
				_,combinedloss_batch = sess.run([train_op_combined, model.combineloss], feed_dict=feed_dict)
				_,domainloss_batch = sess.run([train_op_domain, model.domainloss], feed_dict=feed_dict)
				_,domainfusionloss_batch = sess.run([train_op_domainfusion, model.domainfusionloss], feed_dict=feed_dict)
				# domainloss_batch = 0
				# domainfusionloss_batch = 0


				userloss_epoch += userloss_batch
				itemloss_epoch += itemloss_batch
				reviewloss_epoch += reviewloss_batch
				combinedloss_epoch += combinedloss_batch
				domainloss_epoch += domainloss_batch
				domainfusionloss_epoch += domainfusionloss_batch

			print('epoch: ', epoch + 1, ' userloss: ', userloss_epoch, ' itemloss: ', itemloss_epoch, ' reviewloss: ', reviewloss_epoch, '\ncombinedloss: ', combinedloss_epoch, 'domainloss:', domainloss_epoch, 'domainfusionloss:', domainfusionloss_epoch)
			end = datetime.now()
			print('time = '+str((end-start).seconds))
		embedpath = basedir +'emb2/'
		if (not os.path.exists(embedpath)):
			os.mkdir(embedpath)

		file_flat=open(embedpath +'embed(pt_ft)_Epoch' + str(
			config.num_epoch) +'Batch'+str(config.batch_size) +'_Tflat-addcl(transeBoth-maxmargin(st))('+trainmode+')_addDL(uniform0.1)(trainTwo)-loc(2).txt', 'wb')

		file_sim_ru = open(embedpath + 'embed(pt_ft)_Epoch' + str(
			config.num_epoch) + 'Batch' + str(config.batch_size) + '_TconcatSim(removeU)-addcl(transeBoth-maxmargin(st))('+trainmode+')_addDL(uniform0.1)(trainTwo)-loc(2).txt', 'wb')

		file_sim_all = open(embedpath + 'embed(pt_ft)_Epoch' + str(
			config.num_epoch) + 'Batch' + str(
			config.batch_size) + '_TconcatAllAtt(removeU)-addcl(transeBoth-maxmargin(st))(' + trainmode + ')_addDL(uniform0.1)(trainTwo)-loc.txt', 'wb')

		batches=data.generate_batches(mode='add',trainmode=trainmode)
		num_batch=len(batches)
		embed=[[] for _ in range(data.num_nodes)]
		# embed_sim = [[] for _ in range(data.num_nodes)]
		# embed_sim_dsim = [[] for _ in range(data.num_nodes)]
		embed_sim_ru = [[] for _ in range(data.num_nodes)]
		embed_sim_att = [[] for _ in range(data.num_nodes)]
		for i in range(num_batch):
			batch=batches[i]

			textnode, user, item, user_loc, item_loc, review_rating, item_rating, user_jdate, review_date, item_price, item_wifi, item_acc, item_web, item_phone \
				, text_neg_node, user_neg, item_neg, user_loc_neg, item_loc_neg, review_rating_neg, item_rating_neg, user_jdate_neg, review_date_neg, item_price_neg, item_wifi_neg, item_acc_neg, item_web_neg, item_phone_neg = zip(
				*batch)
			textnode, user, item, user_loc, item_loc, review_rating, item_rating, user_jdate, review_date, item_price, item_wifi, item_acc, item_web, item_phone \
				, text_neg_node, user_neg, item_neg, user_loc_neg, item_loc_neg, review_rating_neg, item_rating_neg, user_jdate_neg, review_date_neg, item_price_neg, item_wifi_neg, item_acc_neg, item_web_neg, item_phone_neg = \
				np.array(textnode), np.array(user), np.array(item), np.array(user_loc), np.array(
					item_loc), np.array(review_rating), np.array(item_rating), np.array(user_jdate), np.array(
					review_date), np.array(item_price), np.array(item_wifi), np.array(item_acc), np.array(
					item_web), np.array(item_phone) \
					, np.array(text_neg_node), np.array(user_neg), np.array(item_neg), np.array(
					user_loc_neg), np.array(item_loc_neg), np.array(review_rating_neg), np.array(
					item_rating_neg), np.array(user_jdate_neg), np.array(review_date_neg), np.array(
					item_price_neg), np.array(item_wifi_neg), np.array(item_acc_neg), np.array(
					item_web_neg), np.array(item_phone_neg)
			textnode = list(map(int, textnode))
			text_neg_node = list(map(int, text_neg_node))
			text, text_neg = data.text[textnode], data.text[text_neg_node]
			domain = data.domain[textnode]
			uniform = []
			for j in range(len(batch)):
				uniform.append([0.5, 0.5])
			uniform = np.array(uniform)

			feed_dict = {
				model.Text: text,
				model.User: user,
				model.Item: item,
				model.User_loc: user_loc,
				model.Item_loc: item_loc,
				model.Review_rating: review_rating,
				model.Item_rating: item_rating,
				model.User_jdate: user_jdate,
				model.Review_date: review_date,
				model.Item_price: item_price,
				model.Item_wifi: item_wifi,
				model.Item_web: item_web,
				model.Item_phone: item_phone,
				model.Item_acc: item_acc,
				model.Text_neg: text_neg,
				model.User_neg: user_neg,
				model.Item_neg: item_neg,
				model.User_loc_neg: user_loc_neg,
				model.Item_loc_neg: item_loc_neg,
				model.Review_rating_neg: review_rating_neg,
				model.Item_rating_neg: item_rating_neg,
				model.User_jdate_neg: user_jdate_neg,
				model.Review_date_neg: review_date_neg,
				model.Item_price_neg: item_price_neg,
				model.Item_wifi_neg: item_wifi_neg,
				model.Item_acc_neg: item_acc_neg,
				model.Item_web_neg: item_web_neg,
				model.Item_phone_neg: item_phone_neg,
				model.Domain: domain,
				model.dropout_keep_prob: 1.0,
				model.Uniform: uniform
			}

			print("current batch = "+ str(i))
			# run the graph 
			convA = sess.run([model.convT,model.T_concat_all_ru,model.T_concat_all_att],feed_dict=feed_dict)
			for i in range(config.batch_size):
				#em=list(convA[i])+list(TA[i])
				print(str(i) +"/" + str(config.batch_size))
				em = list(convA[0][i])
				embed[textnode[i]].append(em)

				em_sim_ru = list(convA[1][i])
				embed_sim_ru[textnode[i]].append(em_sim_ru)
				em_sim_att = list(convA[2][i])
				embed_sim_att[textnode[i]].append(em_sim_att)
		claIDs = data.trainCla_id
		claIDs.extend(data.test_id)
		print('classifier need date len = ' + str(len(claIDs)))
		for train_index in claIDs:
			file_flat.write(str(train_index).encode() + " ".encode())  #打印id
			if embed[train_index]:
				tmp = np.sum(embed[train_index], axis=0) / len(embed[train_index])
				file_flat.write(' '.join(map(str, tmp)).encode() + '\n'.encode())
			else:
				file_flat.write('\n'.encode())

			file_sim_ru.write(str(train_index).encode() + " ".encode())
			if embed_sim_ru[train_index]:
				tmp = np.sum(embed_sim_ru[train_index], axis=0) / len(embed_sim_ru[train_index])
				file_sim_ru.write(' '.join(map(str, tmp)).encode() + '\n'.encode())
			else:
				file_sim_ru.write('\n'.encode())

			file_sim_all.write(str(train_index).encode() + " ".encode())
			if embed_sim_att[train_index]:
				tmp = np.sum(embed_sim_att[train_index], axis=0) / len(embed_sim_att[train_index])
				file_sim_all.write(' '.join(map(str, tmp)).encode() + '\n'.encode())
			else:
				file_sim_all.write('\n'.encode())
