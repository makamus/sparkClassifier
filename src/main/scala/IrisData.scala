import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lxb on 2017/11/8.
  */
object IrisData {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    /**用户数据****/
    val rawData = sc.textFile("iris.txt")
    //val rawData5 = sc.textFile("ml-100k/train_noheader5.tsv")
    val records = rawData.map(line => line.split(","))
    records.first

    val data = records.map( r => {
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(0, r.size - 1).map {
        d =>
          //println("" + d)
          if (d == "?") {
            //println(d)
            0.0
          } else {
            d.toDouble
          }
      }
      LabeledPoint(label, Vectors.dense(features))
     }
    )
    val randomForestModel =  RandomForest.trainClassifier(data,3,Map[Int,Int](),5,"auto","gini",5,32)
    /***创建训练分类模型**/
    //设置逻辑回归和SVM迭代次数
    val numIterations = 10
    //决策树最大深度
    val maxTreeDepth = 5
    //val svmModel = SVMWithSGD.train(data, numIterations)

    //val dtModel = DecisionTree.train(data, Algo.Classification, Gini, maxTreeDepth)
    ///创建决策树
    val predictionsDtModel = randomForestModel.predict(data.map(lp => lp.features))
    predictionsDtModel.foreach( lp => {
      println(lp)
    })
    println("===============================================")
    //data.foreach( lp => println(lp.label))
    val labelAndPreds = data.map(lp => {
      val prediction = randomForestModel.predict(lp.features)
      (lp.label,prediction)
    })
    val testErr =  labelAndPreds.filter( X => X._1!=X._2).count().toDouble/data.count()
    println("Test Error = " + testErr)
  }
}
