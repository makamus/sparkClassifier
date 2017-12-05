import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SQLContext

/**
  * Created by lxb on 2017/12/4.
  */
object MultilayerPerceptronCExamp {

  def main(args: Array[String]) {
    if (args.length < 1) {
      println("Usage:ClassificationPipeline inputDataFile")
      sys.exit(1)
    }
    val conf = new SparkConf().setAppName("Classification with ML Pipeline")
    val sc = new SparkContext(conf)
    val sqlCtx = new SQLContext(sc)

    /** Step 1
      * Read the source data file and convert it to be a dataframe with columns named.
      * 3.6216,8.6661,-2.8073,-0.44699,0
      * 4.5459,8.1674,-2.4586,-1.4621,0
      * 3.866,-2.6383,1.9242,0.10645,0
      * 3.4566,9.5228,-4.0112,-3.5944,0
      * 0.32924,-4.4552,4.5718,-0.9888,0
      * ... ...
      */
    val parsedRDD = sc.textFile(args(0)).map(_.split(",")).map(eachRow => {
      val a = eachRow.map(x => x.toDouble)
      (a(0), a(1), a(2), a(3), a(4))
    })
    val df = sqlCtx.createDataFrame(parsedRDD).toDF(
      "f0", "f1", "f2", "f3", "label").cache()

    /** *
      * Step 2
      * StringIndexer encodes a string column of labels
      * to a column of label indices. The indices are in [0, numLabels),
      * ordered by label frequencies.
      * This can help detect label in raw data and give it an index automatically.
      * So that it can be easily processed by existing spark machine learning algorithms.
      * */
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    /**
      * Step 3
      * Define a VectorAssembler transformer to transform source features data to be a vector
      * This is helpful when raw input data contains non-feature columns, and it is common for
      * such a input data file to contain columns such as "ID", "Date", etc.
      */
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("f0", "f1", "f2", "f3"))
      .setOutputCol("featureVector")


    val layers = Array[Int](4,3,2,2)
    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(512)
      .setSeed(1234L)
      .setMaxIter(32)
      .setFeaturesCol("featureVector")
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    val pipeline = new Pipeline().setStages(Array(labelIndexer,vectorAssembler,mlpc,labelConverter))
    val model = pipeline.fit(trainingData)

    val predictionResultDF = model.transform(testData)

    /**
      * Step 9
      * Select features,label,and predicted label from the DataFrame to display.
      * We only show 20 rows, it is just for reference.
      */
    predictionResultDF.select("f0","f1","f2","f3","label","predictedLabel").show(200)

    /**
      * Step 10
      * The evaluator code is used to compute the prediction accuracy, this is
      * usually a valuable feature to estimate prediction accuracy the trained model.
      */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    println("Testing Error = " + (1.0 - predictionAccuracy))
  }
}
