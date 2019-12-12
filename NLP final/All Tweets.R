api_key <- 'OtgCiwkSUOZSWfzQEOikXWCZ4'
api_secret <- 'ib6QiWKZaCSKf1dHo1c1f34lGfFnCSys9COprmgjcrz087DrkQ'
access_token <- '1178855384288641024-DXojqllXxBUTEMcBwHqNaNnrRY0F8o'
access_token_secret<- 'tkSiUU1jrW2GyaxiaAK9YFvqH6t23lZvJOt8anYhHsBSk'

install.packages("twitteR")
library(twitteR)
#twittR does not include the capability of getting tweets >140 characters
install.packages("rtweet")
library(rtweet)



setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret)


#Getting tweets
tweets <- search_tweets("sexual abuse -filter:retweets", n = 10000, lang = 'en', tweet_mode= 'extended')
tweets_assault <- search_tweets("sexual assault -filter:retweets",n=10000, lang='en', tweet_mode='extended')
#no_retweets <- strip_retweets(tweets)
tweetsdf <- tweets_data(tweets)
write_as_csv(x = tweetsdf, file_name = "C:/Users/PG/Desktop/NLP final/tweets_abuse.csv")
tweetsdf1 <- tweets_data(tweets_assault)
write_as_csv(x = tweetsdf1, file_name = "C:/Users/PG/Desktop/NLP final/tweets_assault.csv")

#Merging the 2 dataframes
all_tweets<- rbind(tweetsdf, tweetsdf1)
#tweetsdf <- tweetsdf[-c(2,5)]
write_as_csv(x = all_tweets, file_name = "C:/Users/PG/Desktop/NLP final/alltweets.csv")


