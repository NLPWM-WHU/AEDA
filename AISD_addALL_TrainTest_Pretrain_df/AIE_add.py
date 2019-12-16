# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from AISD_addALL_TrainTest_Pretrain_df import config


class Model:
    def __init__(self,vocab_size,num_users,num_items,num_locs,num_ratings,num_dates,num_prices,num_wifis,num_accs,num_webs,num_phones,word_matrix):
            
            # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            #review
            self.Text=tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Text')
            self.Text_neg=tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Text_neg')

            #user
            self.User=tf.placeholder(tf.int32, [config.batch_size], name='User')
            self.User_neg=tf.placeholder(tf.int32, [config.batch_size], name='User_neg')

            #item
            self.Item=tf.placeholder(tf.int32, [config.batch_size], name='Item')
            self.Item_neg=tf.placeholder(tf.int32, [config.batch_size], name='Item_neg')

            #user attribute: location
            self.User_loc = tf.placeholder(tf.int32, [config.batch_size], name='User_loc')
            self.User_loc_neg=tf.placeholder(tf.int32, [config.batch_size], name='User_loc_neg')

            self.User_jdate = tf.placeholder(tf.int32, [config.batch_size], name='User_jdate')
            self.User_jdate_neg = tf.placeholder(tf.int32, [config.batch_size], name='User_jdate_neg')


            #item attribute: location rating
            self.Item_loc = tf.placeholder(tf.int32, [config.batch_size], name='Item_loc')
            self.Item_loc_neg=tf.placeholder(tf.int32, [config.batch_size], name='Item_loc_neg')

            self.Item_rating = tf.placeholder(tf.int32, [config.batch_size], name='Item_rating')
            self.Item_rating_neg=tf.placeholder(tf.int32, [config.batch_size], name='Item_rating_neg')

            self.Item_price = tf.placeholder(tf.int32, [config.batch_size], name='Item_price')
            self.Item_price_neg = tf.placeholder(tf.int32, [config.batch_size], name='Item_price_neg')

            self.Item_wifi = tf.placeholder(tf.int32, [config.batch_size], name='Item_wifi')
            self.Item_wifi_neg = tf.placeholder(tf.int32, [config.batch_size], name='Item_wifi_neg')

            self.Item_acc = tf.placeholder(tf.int32, [config.batch_size], name='Item_acc')
            self.Item_acc_neg = tf.placeholder(tf.int32, [config.batch_size], name='Item_acc_neg')

            self.Item_web = tf.placeholder(tf.int32, [config.batch_size], name='Item_web')
            self.Item_web_neg = tf.placeholder(tf.int32, [config.batch_size], name='Item_web_neg')

            self.Item_phone = tf.placeholder(tf.int32, [config.batch_size], name='Item_phone')
            self.Item_phone_neg = tf.placeholder(tf.int32, [config.batch_size], name='Item_phone_neg')

            #review attribute: rating
            self.Review_rating = tf.placeholder(tf.int32, [config.batch_size], name='Review_rating')
            self.Review_rating_neg=tf.placeholder(tf.int32, [config.batch_size], name='Review_rating_neg')

            self.Review_date = tf.placeholder(tf.int32, [config.batch_size], name='Review_date')
            self.Review_date_neg = tf.placeholder(tf.int32, [config.batch_size], name='Review_date_neg')
            #

            self.Domain = tf.placeholder(tf.int32, [config.batch_size, 1], name="domain")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            self.Uniform = tf.placeholder(tf.float32, [config.batch_size, config.DOMAIN_LABEL], name='Uniform')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed=tf.Variable(initial_value=tf.cast(tf.constant(np.array(word_matrix)),tf.float32),expected_shape= (vocab_size+1,config.embed_size),dtype=tf.float32,trainable=True)
            self.text_embed1 = self.text_embed
            #self.text_embed1 = tf.clip_by_norm(self.text_embed, clip_norm=1, axes=1)
            #考虑一下为什么对text_emb不做规范化
            self.user_embed=tf.Variable(tf.truncated_normal([num_users+ 1, config.embed_size], stddev=0.3))
            self.user_embed1=tf.clip_by_norm(self.user_embed,clip_norm=1,axes=1)

            self.item_embed=tf.Variable(tf.truncated_normal([num_items+ 1, config.embed_size], stddev=0.3))
            self.item_embed1=tf.clip_by_norm(self.item_embed,clip_norm=1,axes=1)

            self.loc_embed=tf.Variable(tf.truncated_normal([num_locs+ 1, config.embed_size], stddev=0.3))
            self.loc_embed1=tf.clip_by_norm(self.loc_embed,clip_norm=1,axes=1)

            self.rating_embed=tf.Variable(tf.truncated_normal([num_ratings+ 1, config.embed_size], stddev=0.3))
            self.rating_embed1=tf.clip_by_norm(self.rating_embed,clip_norm=1,axes=1)

            self.date_embed = tf.Variable(tf.truncated_normal([num_dates+ 1, config.embed_size], stddev=0.3))
            self.date_embed1 = tf.clip_by_norm(self.date_embed, clip_norm=1, axes=1)

            self.price_embed = tf.Variable(tf.truncated_normal([num_prices+ 1, config.embed_size], stddev=0.3))
            self.price_embed1 = tf.clip_by_norm(self.price_embed, clip_norm=1, axes=1)

            self.wifi_embed = tf.Variable(tf.truncated_normal([num_wifis+ 1, config.embed_size], stddev=0.3))
            self.wifi_embed1 = tf.clip_by_norm(self.wifi_embed, clip_norm=1, axes=1)

            self.acc_embed = tf.Variable(tf.truncated_normal([num_accs + 1, config.embed_size], stddev=0.3))
            self.acc_embed1 = tf.clip_by_norm(self.acc_embed, clip_norm=1, axes=1)

            self.web_embed = tf.Variable(tf.truncated_normal([num_webs+1, config.embed_size], stddev=0.3))
            self.web_embed1 = tf.clip_by_norm(self.web_embed, clip_norm=1, axes=1)

            self.phone_embed = tf.Variable(tf.truncated_normal([num_phones+1, config.embed_size], stddev=0.3))
            self.phone_embed1 = tf.clip_by_norm(self.phone_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            #review embedding
            self.T=tf.nn.embedding_lookup(self.text_embed1,self.Text)
            self.T_emb=tf.expand_dims(self.T,-1)
           
            self.TNEG=tf.nn.embedding_lookup(self.text_embed1,self.Text_neg)
            self.T_NEG_emb=tf.expand_dims(self.TNEG,-1)

            #user embedding
            self.U=tf.nn.embedding_lookup(self.user_embed1,self.User)
            self.U_NEG=tf.nn.embedding_lookup(self.user_embed1,self.User_neg)

            #item embedding
            self.I=tf.nn.embedding_lookup(self.item_embed1,self.Item)
            self.I_NEG=tf.nn.embedding_lookup(self.item_embed1,self.Item_neg)

            #user location
            self.Uloc=tf.nn.embedding_lookup(self.loc_embed1,self.User_loc)
            self.Uloc_NEG=tf.nn.embedding_lookup(self.loc_embed1,self.User_loc_neg)

            #item location
            self.Iloc=tf.nn.embedding_lookup(self.loc_embed1,self.Item_loc)
            self.Iloc_NEG=tf.nn.embedding_lookup(self.loc_embed1,self.Item_loc_neg)

            #review rating
            self.Rrating=tf.nn.embedding_lookup(self.rating_embed1,self.Review_rating)
            self.Rrating_NEG=tf.nn.embedding_lookup(self.rating_embed1,self.Review_rating_neg)

            #item rating
            self.Irating=tf.nn.embedding_lookup(self.rating_embed1,self.Item_rating)
            self.Irating_NEG=tf.nn.embedding_lookup(self.rating_embed1,self.Item_rating_neg)

            #user joindate
            self.Ujdate=tf.nn.embedding_lookup(self.date_embed1,self.User_jdate)
            self.Ujdate_NEG = tf.nn.embedding_lookup(self.date_embed1,self.User_jdate_neg)

            #review date
            self.Rdate=tf.nn.embedding_lookup(self.date_embed1,self.Review_date)
            self.Rdate_NEG=tf.nn.embedding_lookup(self.date_embed1,self.Review_date_neg)

            #item price
            self.Iprice=tf.nn.embedding_lookup(self.price_embed1,self.Item_price)
            self.Iprice_NEG = tf.nn.embedding_lookup(self.price_embed1,self.Item_price_neg)

            #item wifi
            self.Iwifi=tf.nn.embedding_lookup(self.wifi_embed1,self.Item_wifi)
            self.Iwifi_NEG = tf.nn.embedding_lookup(self.wifi_embed1, self.Item_wifi_neg)

            #item acc
            self.Iacc=tf.nn.embedding_lookup(self.acc_embed1,self.Item_acc)
            self.Iacc_NEG = tf.nn.embedding_lookup(self.acc_embed1, self.Item_acc_neg)

            #item web
            self.Iweb=tf.nn.embedding_lookup(self.web_embed1,self.Item_web)
            self.Iweb_NEG = tf.nn.embedding_lookup(self.web_embed1, self.Item_web_neg)

            #item phone
            self.Iphone=tf.nn.embedding_lookup(self.phone_embed1,self.Item_phone)
            self.Iphone_NEG = tf.nn.embedding_lookup(self.phone_embed1, self.Item_phone_neg)

        self.convT,self.convTNEG,self.T_concat_all_ru,self.T_concat_all_att=self.conv()
        self.userloss = self.compute_loss('UserView')
        self.itemloss = self.compute_loss('ItemView')
        self.reviewloss = self.compute_loss('ReviewView')
        self.combineloss = self.compute_loss('Combined')
        self.domainloss = self.compute_loss('Domain')
        self.domainfusionloss = self.compute_loss('Domainfusion')


    def conv(self):
        self.W2=tf.Variable(tf.truncated_normal([2, config.embed_size, 1, config.embed_size], stddev=0.3))
        self.b = tf.Variable(tf.constant(0.0, shape=[config.embed_size]), name="b")
        convT=tf.nn.conv2d(self.T_emb,self.W2,strides=[1,1,1,1],padding='VALID')
        convT= tf.nn.tanh(tf.nn.bias_add(convT, self.b))
        convTNEG=tf.nn.conv2d(self.T_NEG_emb,self.W2,strides=[1,1,1,1],padding='VALID')
        convTNEG = tf.nn.tanh(tf.nn.bias_add(convTNEG, self.b))

        pooled_T=tf.nn.max_pool(convT,ksize=[1,config.MAX_LEN - 2 +1,1,1],
            strides=[1,1,1,1],padding="VALID",name="pool")
        pooled_TNEG=tf.nn.max_pool(convTNEG,ksize=[1,config.MAX_LEN - 2 +1,1,1],
            strides=[1,1,1,1],padding="VALID",name="pool")
        # pooled_T=tf.reduce_mean(convT,1)
        # pooled_TNEG=tf.reduce_mean(convTNEG,1)

        T_flat=tf.squeeze(pooled_T)
        TNEG_flat=tf.squeeze(pooled_TNEG)

        #
        # uisim = tf.expand_dims(tf.reduce_sum(tf.multiply(self.U,self.I), 1),1)
        # utsim = tf.expand_dims(tf.reduce_sum(tf.multiply(self.U,T_flat), 1),1)
        # itsim = tf.expand_dims(tf.reduce_sum(tf.multiply(self.I,T_flat), 1),1)

        # locsim = tf.expand_dims(tf.reduce_sum(tf.multiply(self.Uloc,self.Iloc), 1),1)
        # ratingsim =tf.expand_dims(tf.reduce_sum(tf.multiply(self.Rrating,self.Irating), 1),1)
        #datesim = tf.expand_dims(tf.reduce_sum(tf.multiply(self.Ujdate, self.Rdate), 1), 1)

        # locsim = self.Uloc - self.Iloc
        # ratingsim = self.Rrating - self.Irating
        #T_concat_noText = tf.concat([self.Ujdate,locsim,ratingsim], axis=1)
        # T_concat_all = tf.concat([T_flat, self.Ujdate,locsim, ratingsim, uisim, utsim, itsim], axis=1)
        # T_concat_all_dsim = tf.concat([T_flat, self.Ujdate,locsim, ratingsim, uisim, utsim, itsim], axis=1)
        T_concat_all_ru = tf.concat([T_flat,self.Rrating,self.Uloc,self.Ujdate],axis=1)
        T_concat_all_att = tf.concat(
            [T_flat,self.Rrating,self.Uloc,self.Ujdate,self.Iacc,self.Iwifi,self.Iprice,self.Iweb,self.Iphone],
            axis=1)
        # T_concat_all_ru = T_flat + self.Uloc + self.Rrating + self.Ujdate
        # T_concat_all_att = T_flat + self.Uloc + self.Rrating + self.Ujdate + self.Iprice + self.Iwifi + self.Iacc + self.Iweb + self.Iphone
        # for domain classifer
        # Add dropout
        self.h_drop = tf.nn.dropout(T_concat_all_att, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions


        self.Wdomain = tf.get_variable(
            "Wdomain",
            shape=[900, config.DOMAIN_LABEL],
            initializer=tf.contrib.layers.xavier_initializer())
        self.bdomain = tf.Variable(tf.constant(0.0, shape=[config.DOMAIN_LABEL]), name="bdomain")

        self.scores = tf.nn.xw_plus_b(self.h_drop, self.Wdomain, self.bdomain, name="scores")


        return T_flat,TNEG_flat,T_concat_all_ru,T_concat_all_att
    def compute_loss(self,lossType):

        if lossType == 'UserView':
            p1 = tf.reduce_sum(tf.multiply(self.U,self.Uloc),1)
            p1=tf.log(tf.sigmoid(p1)+0.001)
            p2=tf.reduce_sum(tf.multiply(self.U,self.Uloc_NEG),1)
            p2=tf.log(tf.sigmoid(-p2)+0.001)
            #
            p15 = tf.reduce_sum(tf.multiply(self.U,self.Ujdate),1)
            p15 = tf.log(tf.sigmoid(p15) + 0.001)
            p16 = tf.reduce_sum(tf.multiply(self.U,self.Ujdate_NEG),1)
            p16 = tf.log(tf.sigmoid(-p16) + 0.001)

            temp_loss =  p1+p2+p15 + p16

        elif lossType == 'ItemView':
            p3=tf.reduce_sum(tf.multiply(self.I,self.Iloc),1)
            p3=tf.log(tf.sigmoid(p3)+0.001)
            p4=tf.reduce_sum(tf.multiply(self.I,self.Iloc_NEG),1)
            p4=tf.log(tf.sigmoid(-p4)+0.001)

            p5 = tf.reduce_sum(tf.multiply(self.I, self.Irating), 1)
            p5 = tf.log(tf.sigmoid(p5) + 0.001)
            p6 = tf.reduce_sum(tf.multiply(self.I, self.Irating_NEG), 1)
            p6 = tf.log(tf.sigmoid(-p6) + 0.001)

            p17 = tf.reduce_sum(tf.multiply(self.I, self.Iprice), 1)
            p17 = tf.log(tf.sigmoid(p17) + 0.001)
            p18 = tf.reduce_sum(tf.multiply(self.I, self.Iprice_NEG), 1)
            p18 = tf.log(tf.sigmoid(-p18) + 0.001)

            p19 = tf.reduce_sum(tf.multiply(self.I, self.Iwifi), 1)
            p19 = tf.log(tf.sigmoid(p19) + 0.001)
            p20 = tf.reduce_sum(tf.multiply(self.I, self.Iwifi_NEG), 1)
            p20 = tf.log(tf.sigmoid(-p20) + 0.001)

            p21 = tf.reduce_sum(tf.multiply(self.I, self.Iacc), 1)
            p21 = tf.log(tf.sigmoid(p21) + 0.001)
            p22 = tf.reduce_sum(tf.multiply(self.I, self.Iacc_NEG), 1)
            p22 = tf.log(tf.sigmoid(-p22) + 0.001)

            p25 = tf.reduce_sum(tf.multiply(self.I, self.Iweb), 1)
            p25 = tf.log(tf.sigmoid(p25) + 0.001)
            p26 = tf.reduce_sum(tf.multiply(self.I, self.Iweb_NEG), 1)
            p26 = tf.log(tf.sigmoid(-p26) + 0.001)

            p27 = tf.reduce_sum(tf.multiply(self.I, self.Iphone), 1)
            p27 = tf.log(tf.sigmoid(p27) + 0.001)
            p28 = tf.reduce_sum(tf.multiply(self.I, self.Iphone_NEG), 1)
            p28 = tf.log(tf.sigmoid(-p28) + 0.001)

            #temp_loss = p3 + p4 + p5 + p6 + p19 + p20 + p21 + p22 + p25 + p26 + p27 + p28
            temp_loss = p3+p4+p5+p6+p17+p18+p19+p20+p21+p22+ p25+ p26+p27+p28

        elif lossType == 'ReviewView':

            p7=tf.reduce_sum(tf.multiply(self.convT,self.Rrating),1)
            p7=tf.log(tf.sigmoid(p7)+0.001)
            p8=tf.reduce_sum(tf.multiply(self.convT,self.Rrating_NEG),1)
            p8=tf.log(tf.sigmoid(-p8)+0.001)

            p23=tf.reduce_sum(tf.multiply(self.convT,self.Rdate),1)
            p23=tf.log(tf.sigmoid(p23)+0.001)
            p24=tf.reduce_sum(tf.multiply(self.convT,self.Rdate_NEG),1)
            p24=tf.log(tf.sigmoid(-p24)+0.001)

            #temp_loss = p23 + p24
            temp_loss = p7+p8+p23 + p24


        elif lossType == 'Combined':
            # p9=tf.reduce_sum(tf.multiply(self.U,self.convT),1)
            # p9=tf.log(tf.sigmoid(p9)+0.001)
            # p10=tf.reduce_sum(tf.multiply(self.U,self.convTNEG),1)
            # p10=tf.log(tf.sigmoid(-p10)+0.001)
            #
            # p11=tf.reduce_sum(tf.multiply(self.convT,self.I),1)
            # p11=tf.log(tf.sigmoid(p11)+0.001)
            # p12=tf.reduce_sum(tf.multiply(self.convT,self.I_NEG),1)
            # p12=tf.log(tf.sigmoid(-p12)+0.001)
            #
            # p13 = tf.reduce_sum(tf.multiply(self.I, self.U), 1)
            # p13 = tf.log(tf.sigmoid(p13) + 0.001)
            # p14 = tf.reduce_sum(tf.multiply(self.I, self.U_NEG), 1)
            # p14 = tf.log(tf.sigmoid(-p14) + 0.001)
            #
            # temp_loss = p9 + p10 + p11+ p12+p13 + p14


            pos = tf.reduce_sum(tf.square(self.I/self.getL1(self.I) +self.U/self.getL1(self.U) - self.convT/self.getL1(self.convT)), 1)
            neg1 = tf.reduce_sum(tf.square(
                self.I_NEG / self.getL1(self.I_NEG) + self.U / self.getL1(self.U) - self.convT / self.getL1(self.convT)), 1)

            neg2 = tf.reduce_sum(tf.square(
                self.I / self.getL1(self.I) + self.U / self.getL1(self.U) - self.convTNEG / self.getL1(self.convTNEG)), 1)

            p32 = -tf.reduce_sum(tf.maximum(pos - neg1 + 1, 0))
            p33 = -tf.reduce_sum(tf.maximum(pos - neg2 + 1, 0))
            temp_loss = p32+p33

        elif lossType == 'Domain':
            Domainlabel = tf.squeeze(self.Domain)
            temp_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels =Domainlabel, logits = self.scores)

        elif lossType == 'Domainfusion': #领域混淆损失
            temp_loss = -tf.nn.softmax_cross_entropy_with_logits(labels = self.Uniform, logits = self.scores)
            # Domainlabel = tf.squeeze(self.Domain)
            # temp_loss = -0.1*tf.nn.sparse_softmax_cross_entropy_with_logits(labels= 1-Domainlabel, logits=self.scores)

        else:

            print('loss error')
        loss=-tf.reduce_sum(temp_loss)
        return loss
    def getL1(self,input):
        result =tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(input),1)), -1)
        return result
