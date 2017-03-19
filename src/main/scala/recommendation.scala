/**
  * Created by zhangluoyang on 17-3-18.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD._

object recommendation {

  val PRIOR_COUNT = 10
  val PRIOR_CORRELATION = 0


  def correlation(size: Double, dotProduct: Double, ratingSum: Double,
                  rating2Sum: Double, ratingNormsq: Double, rating2NormSq: Double) = {
    val numerator = size*dotProduct - ratingSum*rating2Sum
    val denominator = math.sqrt(size*ratingNormsq-ratingSum*ratingSum)*
    math.sqrt(size*rating2NormSq - rating2Sum*rating2Sum)
    numerator / denominator
  }

  def regularizedCorrelation(size: Double, dotProduct: Double, ratingSum: Double,
                             rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double,
                             virtualCount: Double, priorCorrelation: Double)={
    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size+virtualCount)
    w*unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  def cosineSimilarity(dotProduct: Double, ratingNorm: Double, rating2Norm: Double)={
    dotProduct / (ratingNorm*rating2Norm)
  }

  def jaccardSimilarity(usersInCommon: Double, totalUsers1 : Double, totalUsers2 : Double)={
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon/ union
  }

  def list2str(list: List[Long])={
    var s = ""
    for (d <- list){
    s += ", " + d
  }
    s.substring(2, s.length)
  }

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("collaborative filtering recommendation based on items").setMaster("local[4]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(sparkConf)

    val users_items_ratings = sc.textFile("ratings.dat", 4).map {
      line =>
        val line_split = line.split("::")
        // user_id, item_id, rating
        (line_split(0).toLong, line_split(1).toLong, line_split(2).toDouble)
    }
    val users_items_map = users_items_ratings.keyBy(x=>x._1).groupByKey().map(x=>(x._1, x._2.map(t=>t._2).toList)).collectAsMap()
    println(!users_items_map(23).contains(0))
    // calculate the numbers of users for each item_id
    val num_users_each_item = users_items_ratings.groupBy(x=>x._2).map(x=>(x._1, x._2.size))
    // num_users_each_item.collect().foreach(println(_))
    // join users_items_ratings with users_items_rating by item_id   (user_id, item_id, rating, num_users)
    val users_items_ratings_num = users_items_ratings.groupBy(x=>x._2).join(num_users_each_item).flatMap(x=>{
      x._2._1.map(t=>(t._1, t._2, t._3, x._2._2))
    })
    val userId_users_items_ratings_num = users_items_ratings_num.keyBy(x=>x._1)

    val rating_pairs = users_items_ratings_num
      .keyBy(x=>x._1)
      .join(userId_users_items_ratings_num)
      .filter(f=>f._2._1._2 < f._2._2._2)
    // some process each pairs
    val vectorCalcs = rating_pairs.map(data=>{
      val key = (data._2._1._2, data._2._2._2)
      val stats = (
        data._2._1._3 * data._2._2._3,  // rating 1 * rating2
        data._2._1._3,  // rating movie 1
        data._2._2._3,  // rating movie 2
        math.pow(data._2._1._3, 2),  //  square of score
        math.pow(data._2._2._3, 2),  //
        data._2._1._4,    //  users
        data._2._2._4     //  users
      )
      (key, stats)
    })
      .groupByKey()
      .map(x=>{
        val key = x._1
        val vals = x._2
        val size = vals.size
        val dotProduct = vals.map(t=>t._1).sum
        val ratingSum = vals.map(t=>t._2).sum
        val rating2Sum = vals.map(t=>t._3).sum
        val ratingSq = vals.map(t=>t._4).sum
        val rating2Sq = vals.map(t=>t._5).sum
        val numRaters = vals.map(t=>t._6).max
        val numRaters2 = vals.map(t=>t._7).max
        (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
      })
    val similarities = vectorCalcs.map(
      x=>{
        val key = x._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = x._2
        val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
        val cosSim = cosineSimilarity(dotProduct, math.sqrt(ratingNormSq), math.sqrt(rating2NormSq))
        val jaccard = jaccardSimilarity(size, numRaters, numRaters2)
        (key, (corr, regCorr, cosSim, jaccard))
      })
    // recommendation
    val items_users_items_ratings = users_items_ratings.keyBy(x=>x._2)
    val items_similarities = similarities.map(x=>(x._1._1, (x._1._2, x._2._3)))
    val users_items_ratingsim = items_users_items_ratings.join(items_similarities).map(x=>((x._2._1._1, x._2._2._1), x._2._1._3*x._2._2._2))
    // remove rated items
    val scores = users_items_ratingsim.groupByKey().map(x=>{
      val key = x._1
      val score = x._2.sum
      (key._1,(key._2, score))
    }).groupByKey().map(x=>(x._1, x._2.toList.sortBy(t=>t._2)))
      .map(x=>x._1 :: x._2.map(t=>t._1).filter(p=> users_items_map(x._1).contains(p)==false).take(10)).map(x=>list2str(x))
//    println(scores.first())
    scores.saveAsTextFile("result")
  }
}
