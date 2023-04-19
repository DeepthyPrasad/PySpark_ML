# SENTIMENTAL ANALYSIS 'NLP based binary classification using Decision Tree and Random Forest Classifiers'

from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import sys
from functools import reduce

# Function for combining multiple DataFrames row-wise
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

if __name__ == "__main__":
    # Create a SparkContext and an SQLContext
    sc = SparkContext(appName="Sentiment Classification")
    sqlContext = SQLContext(sc)

    # Load data
    # wholeTextFiles(path, [...]) reads a directory of text files from a filesystem
    # Each file is read as a single record and returned in a key-value pair
    # The key is the path and the value is the content of each file
    reviews = sc.wholeTextFiles('hdfs:///file_path'+sys.argv[1]+'/*/*')
    
    reviews_words = reviews.flatMap(lambda line: line[1].split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    reviews_sorted = sorted(reviews_words.collect(), key=lambda x: x[1], reverse = True)
    print("Total number of unique tokens in all documents: " + str(len(reviews_sorted)-1)) 
    print("Ten most frequent tokens and their frequencies: " + str(reviews_sorted[:10]))
    print("Ten least frequent tokens and their frequencies: " + str(reviews_sorted[-10:]))

    # Create tuples: (class label, review text) - we ignore the file path
    # 1.0 for positive reviews
    # 0.0 for negative reviews
    reviews_f = reviews.map(lambda row: (1.0 if 'pos' in row[0] else 0.0, row[1]))

    # Convert data into a Spark SQL DataFrame
    # The first column contains the class label
    # The second column contains the review text
    dataset = reviews_f.toDF(['class_label', 'review'])

    # ----- PART II: FEATURE ENGINEERING -----

    # Tokenize the review text column into a list of words
    tokenizer = Tokenizer(inputCol='review', outputCol='words')
    words_data = tokenizer.transform(dataset)

    # Randomly split data into a training set, a development set and a test set
    (train, dev, test) = words_data.randomSplit([.6, .2, .2], seed = 42)

    # Count the number of instances in, respectively, train, dev and test    
    print("Number of train instances: " + str(train.count()))
    print("Number of dev instances: " + str(dev.count()))
    print("Number of test instances: " + str(test.count()))

    # Count the number of positive/negative instances in, respectively, train, dev and test
    # Print the class distribution for each to standard output and then percentage of the same
    
    train_positive = train.filter(train.class_label == 1)
    print("Train positive count: " + str(train_positive.count()))
    dev_positive = dev.filter(dev.class_label == 1)
    print("Dev positive count: " + str(dev_positive.count()))
    test_positive = test.filter(test.class_label == 1)
    print("Test positive count: " + str(test_positive.count()))
    
    train_negative = train.filter(train.class_label == 0)
    print("Train negative count: " + str(train_negative.count()))
    dev_negative = dev.filter(dev.class_label == 0)
    print("Dev negative count: " + str(dev_negative.count()))
    test_negative = test.filter(test.class_label == 0)
    print("Test negative count: " + str(test_negative.count()))
    
    train_distribution = (float(train_positive.count())/float(train.count()))*100
    dev_distribution = (float(dev_positive.count())/float(dev.count()))*100
    test_distribution = (float(test_positive.count())/float(test.count()))*100
    
    print("Train distribution: " + str(train_distribution) + " %")
    print("Dev distribution: " + str(dev_distribution) + " %")
    print("Test distribution: " + str(test_distribution) + " %")

    # Create a stopword list containing the 100 most frequent tokens in the training data
    stop_words = train.rdd.flatMap(lambda line: line[1].split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    stop_words = sorted(stop_words.collect(), key=lambda x: x[1], reverse = True)
    list_top100_words = stop_words[:100]
    list_top100_words = [frequency_tuple[0] for frequency_tuple in list_top100_words]

    # Replace the [] in the stopWords parameter with the name of your created list
    remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=list_top100_words)

    # Remove stopwords from all three subsets
    train_filtered = remover.transform(train)
    dev_filtered = remover.transform(dev)
    test_filtered = remover.transform(test)

    # Transform data to a bag of words representation
    # Only includes tokens that have a minimum document frequency of 2
    cv = CountVectorizer(inputCol='words_filtered', outputCol='BoW', minDF=2.0)
    cv_model = cv.fit(train_filtered)
    train_data = cv_model.transform(train_filtered)
    dev_data = cv_model.transform(dev_filtered)
    test_data = cv_model.transform(test_filtered)
    
    #Print the vocabulary size (to STDOUT) after filtering out stopwords and very rare tokens    
    vocabulary_size = len(cv_model.vocabulary)
    print("Vocabulary size: " + str(vocabulary_size))

    # Create a TF-IDF representation of the data
    idf = IDF(inputCol='BoW', outputCol='TFIDF')
    idf_model = idf.fit(train_data)
    train_tfidf = idf_model.transform(train_data)
    dev_tfidf = idf_model.transform(dev_data)
    test_tfidf = idf_model.transform(test_data)

    # ----- PART III: MODEL SELECTION -----

    # Provide information about class labels: needed for model fitting
    label_indexer = StringIndexer(inputCol = 'class_label', outputCol = 'label')

    # Create an evaluator for binary classification
    evaluator = BinaryClassificationEvaluator()

    # Train a decision tree with default parameters (including maxDepth=5)
    dt_classifier_default = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=5)

    # Create an ML pipeline for the decision tree model
    dt_pipeline_default = Pipeline(stages=[label_indexer, dt_classifier_default])

    # Apply pipeline and train model
    dt_model_default = dt_pipeline_default.fit(train_tfidf)

    # Apply model on devlopment data
    dt_predictions_default_dev = dt_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_dt_default_dev = evaluator.evaluate(dt_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev))

    #Check for signs of overfitting (by evaluating the model on the training set)
    auc_dt_default_train = evaluator.evaluate(dt_model_default.transform(train_tfidf), {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Training Set, AUC:' + str(auc_dt_default_train))

    # Tune the decision tree model by changing one of its hyperparameters - maxDepth values: 3 and 4.    
    dt_classifier_depth_3 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=3)
    dt_classifier_depth_4 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=4)
    
    dt_pipeline_depth_3 = Pipeline(stages=[label_indexer, dt_classifier_depth_3])
    dt_model_depth_3 = dt_pipeline_depth_3.fit(train_tfidf)
    dt_predictions_depth_3_dev = dt_model_depth_3.transform(dev_tfidf)
    auc_dt_3_depth_dev = evaluator.evaluate(dt_predictions_depth_3_dev, {evaluator.metricName: 'areaUnderROC'})
    
    dt_pipeline_depth_4 = Pipeline(stages=[label_indexer, dt_classifier_depth_4])
    dt_model_depth_4 = dt_pipeline_depth_4.fit(train_tfidf)
    dt_predictions_depth_4_dev = dt_model_depth_4.transform(dev_tfidf)
    auc_dt_4_depth_dev = evaluator.evaluate(dt_predictions_depth_4_dev, {evaluator.metricName: 'areaUnderROC'})
    
    print("Decision Tree, Depth Parameters = 3, Development Set, AUC: " + str(auc_dt_3_depth_dev))
    print("Decision Tree, Depth Parameters = 4, Development Set, AUC: " + str(auc_dt_4_depth_dev))

    # Train a random forest with default parameters (including numTrees=20)
    rf_classifier_default = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=20)

    # Create an ML pipeline for the random forest model
    rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

    # Apply pipeline and train model
    rf_model_default = rf_pipeline_default.fit(train_tfidf)

    # Apply model on development data
    rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Random Forest, Default Parameters, Development Set, AUC:' + str(auc_rf_default_dev))

    # Check for signs of overfitting (by evaluating the model on the training set)    
    auc_rf_default_train = evaluator.evaluate(rf_model_default.transform(train_tfidf), {evaluator.metricName: 'areaUnderROC'})
    print("Random Forest, Default Parameters, Training Set, AUC: " + str(auc_rf_default_train))

    # Tune the random forest model by changing one of its hyperparameters
    rf_classifier_num_100 = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)
    rf_pipeline_num_100 = Pipeline(stages=[label_indexer, rf_classifier_num_100])
    rf_model_num_100 = rf_pipeline_num_100.fit(train_tfidf)
    rf_predictions_num_100_dev = rf_model_num_100.transform(dev_tfidf)
    auc_rf_num_100_dev = evaluator.evaluate(rf_predictions_num_100_dev, {evaluator.metricName: 'areaUnderROC'})
    print("Random Forest, Number Trees Parameters = 100, Development Set, AUC: " + str(auc_rf_num_100_dev))

    # ----- PART IV: MODEL EVALUATION -----

    # Create a new dataset combining the train and dev sets
    traindev_tfidf = unionAll(train_tfidf, dev_tfidf)
    
    # Build a new model from the concatenation of the train and dev sets in order to better utilize the data    
    rf_classifier_num_100 = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)
    rf_pipeline_num_100 = Pipeline(stages=[label_indexer, rf_classifier_num_100])
    rf_model_num_100 = rf_pipeline_num_100.fit(traindev_tfidf)
    rf_predictions_num_100_test = rf_model_num_100.transform(test_tfidf)
    auc_rf_num_100_test = evaluator.evaluate(rf_predictions_num_100_test, {evaluator.metricName: 'areaUnderROC'})
    print("Random Forest, Number Trees Parameters = 100, Test Set, AUC: " + str(auc_rf_num_100_test))
