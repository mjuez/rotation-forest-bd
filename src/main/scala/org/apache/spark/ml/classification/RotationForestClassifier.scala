/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.classification

import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonAST.JValue

import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors, Matrix, Matrices}
import org.apache.spark.ml.util._
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.EnsembleModelReadWrite.EnsembleNodeData
import org.apache.spark.ml.param._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{PCA, VectorSlicer}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, DenseVector => OldDenseVector,
  Matrices => OldMatrices, Vector => OldVector, Vectors => OldVectors}
import scala.collection.mutable
import org.apache.spark.ml.feature.{ MinMaxScaler, MinMaxScalerModel }
import scala.util.Random
import org.apache.spark.ml.linalg.SparseMatrix
import org.apache.hadoop.fs.{FileSystem, Path}

/**
 * Rotation Forest shared functionality.
 * Contains column names and parallel instance 
 * rotation implementation.
 * 
 * @author Mario Juez-Gil <mariojg@ubu.es>
 */
@Since("2.4.4")
object RotationForestClassifier {
  
  val SCALED_FEATURE_COL = "sf"
  val PROJECTED_OUTPUT_COL = "pof"
  val SLICED_FEATURES_COL = "slf"
  
  /**
    * Rotates a feature vector given a rotation matrix.
    * The rotation is performed in a parallel way, through
    * Matrix multiplication.
    *
    * @param rotationMatrix Rotation Matrix.
    * @param features Features to rotate.
    */
  @Since("2.4.4")
  def rotate(rotationMatrix: Matrix, features: Vector): Vector = {
    features match {
      case dv: DenseVector =>
        rotationMatrix.transpose.multiply(dv)
      case sv: SparseVector =>
        rotationMatrix.transpose.multiply(sv.toDense)
      case _ =>
        throw new IllegalArgumentException("Unsupported vector format. Expected " +
          s"SparseVector or DenseVector. Instead got: ${features.getClass}")
    }
  }

}

/**
 * Rotation Forest Classifier.
 * Spark port of the well-known Rotation Forest [1] algorithm.
 * 
 * [1] Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). 
 *     Rotation forest: A new classifier ensemble method. IEEE transactions 
 *     on pattern analysis and machine intelligence, 28(10), 1619-1630.
 * 
 * @author Mario Juez-Gil <mariojg@ubu.es>
 * @param uid
 */
@Since("2.4.4")
class RotationForestClassifier @Since("2.4.4") (
    @Since("2.4.4") override val uid: String) 
  extends ProbabilisticClassifier[Vector, RotationForestClassifier, RotationForestClassificationModel]
  with RotationForestClassifierParams with DefaultParamsWritable {

  import RotationForestClassifier._

  @Since("2.4.4")
  def this() = this(Identifiable.randomUID("rotfc"))

  /** @group setParam */
  def setGroupParamAsNumberOfGroups(value: Boolean): this.type = 
    set(groupParamAsNumberOfGroups, value)

  /** @group setParam */
  def setMinGroup(value: Int): this.type = set(minGroup, value)

  /** @group setParam */
  def setMaxGroup(value: Int): this.type = set(maxGroup, value)

  /** @group setParam */
  def setBootstrapSampleSize(value: Double): this.type = 
    set(bootstrapSampleSize, value)

  /** @group setParam */
  def setNumRotations(value: Int): this.type = set(numRotations, value)

  /** @group setParam */
  def setNormalizeData(value: Boolean): this.type = set(normalizeData, value)

  /** @group setParam */
  def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  def setBootstrap(value: Boolean): this.type = set(bootstrap, value)

  /** @group setParam */
  def setSubsamplingRate(value: Double): this.type = 
    set(subsamplingRate, value)

  /** @group setParam */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  def setMinInstancesPerNode(value: Int): this.type = 
    set(minInstancesPerNode, value)

  def setMinWeightFractionPerNode(value: Double): this.type = 
    set(minWeightFractionPerNode, value)

  /** @group setParam */
  def setMinInfoGain(value: Double): this.type = 
    set(minInfoGain, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = 
    set(checkpointInterval, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setExpertParam */
  def setMaxMemoryInMB(value: Int): this.type = 
    set(maxMemoryInMB, value)
  
  /** @group setExpertParam */
  def setCacheNodeIds(value: Boolean): this.type = 
    set(cacheNodeIds, value)

  /**
    * Trains a Rotation Forest classification model.
    *
    * @param dataset Training dataset.
    * @return Rotation Forest classification model.
    */
  @Since("2.4.4")
  protected def train(dataset: Dataset[_]): RotationForestClassificationModel = {
    val categoricalFeatures =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    require(categoricalFeatures.size == 0, "Rotation Forest cannot work with " +
      "categorical features. Consider transforming the categorical features " +
      "using OneHotEncoderEstimator for example.")

    val numClasses = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
      case Some(n: Int) => n
      case None => throw new IllegalArgumentException("The dataset must " +
      "contain a label column with spark ML metadata. See spark ml " +
      "StringIndexer transformer for more information.")
    }

    val rnd = new Random($(seed))
    
    val numFeatures = countFeatures(dataset)
    val classes = 0d to (numClasses - 1) by 1d toList

    val (scaledDataset, scaler) = scaleDatasetAndGetScalerModel(dataset)
    val cachedDataset = scaledDataset.cache()
    val ensemble = buildEnsemble(cachedDataset, numFeatures, classes, rnd)
    cachedDataset.unpersist()
    
    new RotationForestClassificationModel(ensemble, scaler, numClasses)
  }

  /**
    * Ensemble builder. Given a dataset, it is splitted
    * in many feature groups and as PCA models as groups
    * are constructed. All PCA rotation matrices are
    * rearranged into a single rotation matrix for the
    * dataset, then it is rotated, and a Random Forest 
    * is trained with it. The resulting ensemble will be
    * formed by L rotation matrices and L Random Forest models.
    *
    * @param dataset The training dataset.
    * @param numFeatures The number of features.
    * @param classes The number of classes.
    * @param rnd A random seed
    * @return An array of pairs Rotation-Matrix/Array of Decision tree models.
    */
  @Since("2.4.4")
  private def buildEnsemble(
      dataset: Dataset[_], 
      numFeatures: Int, 
      classes: List[Double],
      rnd: Random): Array[(Matrix, Array[DecisionTreeClassificationModel])] = {
    
    val numClasses = classes.length

    (1 to $(numRotations)).map { _ =>
      val featureGroups = generateGroups(numFeatures, rnd)

      val featureFilters = featureGroups.map { featureIndices =>
        val selectedClasses = rnd.shuffle(classes)
          .drop(rnd.nextInt(numClasses))
        
        val dataSubset = dataset
          .filter(col($(labelCol)).isin(selectedClasses:_*))
          .cache()
        val bootstrapSubset = dataSubset
          .sample($(bootstrapSampleSize), rnd.nextInt)
          .cache()
        
        val slicer = new VectorSlicer()
          .setInputCol($(featuresCol))
          .setIndices(featureIndices)
          .setOutputCol(SLICED_FEATURES_COL)
        val pca = new PCA()
          .setInputCol(SLICED_FEATURES_COL)
          .setOutputCol(PROJECTED_OUTPUT_COL)
          .setK(featureIndices.length)
    
        val pcaModel = {
          try {
            val slicedSubset = slicer.transform(bootstrapSubset)
            pca.fit(slicedSubset)
          }catch{
            case _ : Throwable => {
              val slicedSubset = slicer.transform(dataSubset)
              pca.fit(slicedSubset)
            }
          }
        }
        
        bootstrapSubset.unpersist()
        dataSubset.unpersist()

        pcaModel
      }

      var currRow = 0
      val entries = mutable.ArrayBuilder.make[(Int, Int, Double)]
      for(i <- 0 to featureFilters.length - 1){
        val pcaModel = featureFilters(i)
        val featureIndices = featureGroups(i)
        for(row <- 0 to pcaModel.pc.numRows - 1){
          for(col <- 0 to pcaModel.pc.numCols - 1){
            entries += ((featureIndices(col), currRow, pcaModel.pc(col, row)))
          }
          currRow += 1
        }
      }
      val rotationMatrix = SparseMatrix.fromCOO(numFeatures, numFeatures, entries.result())

      val rotator = udf{ features: Vector => rotate(rotationMatrix, features) }

      // cache is important. randomforest does not caches dataset.
      val rotatedDataset = dataset
        .withColumn($(featuresCol), rotator(dataset($(featuresCol)))).cache()
        
      val rfc = new RandomForestClassifier()
        .setLabelCol($(labelCol))
        .setFeaturesCol($(featuresCol))
        .setNumTrees($(numTrees))
        .setBootstrap($(bootstrap))
        .setSubsamplingRate($(subsamplingRate))
        .setMaxDepth($(maxDepth))
        .setMaxBins($(maxBins))
        .setMinInstancesPerNode($(minInstancesPerNode))
        .setMinWeightFractionPerNode($(minWeightFractionPerNode))
        .setMinInfoGain($(minInfoGain))
        .setCheckpointInterval($(checkpointInterval))
        .setSeed($(seed))
        .setMaxMemoryInMB($(maxMemoryInMB))
        .setCacheNodeIds($(cacheNodeIds))
        .setFeatureSubsetStrategy($(featureSubsetStrategy))
        .setLeafCol($(leafCol))
        .fit(rotatedDataset)

      rotatedDataset.unpersist()

      (rotationMatrix, rfc.trees)
    }.toArray
  }

  /**
    * Utility function for counting the number of features
    * in a dataset.
    *
    * @param dataset The dataset.
    * @return The number of features of the dataset.
    */
  @Since("2.4.4")
  private def countFeatures(dataset: Dataset[_]): Int = {
    try {
      dataset
        .schema($(featuresCol))
        .metadata
        .getMetadata("ml_attr")
        .getLong("num_attrs")
        .asInstanceOf[Int]
    }catch{
      case x: NoSuchElementException => {
        dataset
          .schema($(featuresCol))
          .metadata
          .getLong("numFeatures")
          .asInstanceOf[Int]
      }
    }
  }

  /**
    * Trains a scaler model and scales the dataset.
    * All feature values will be between 0 and 1 (normalized).
    * If the user chosed not to normalize, this function returns
    * the dataset as it was.
    *
    * @param dataset Dataset to scale.
    * @return A pair formed by the scaled dataset and the scaler model.
    */
  @Since("2.4.4")
  private def scaleDatasetAndGetScalerModel(dataset: Dataset[_]): (DataFrame, Option[MinMaxScalerModel]) = {
    if($(normalizeData)){
      val minMaxScaler = new MinMaxScaler()
        .setInputCol($(featuresCol))
        .setOutputCol(SCALED_FEATURE_COL)
      val model = minMaxScaler.fit(dataset)
      val scaledDataSet = model.transform(dataset)
        .select(SCALED_FEATURE_COL, $(labelCol))
        .withColumnRenamed(SCALED_FEATURE_COL, $(featuresCol))
      (scaledDataSet, Some(model))
    } else {
      (dataset.toDF, None)
    }
  }

  /**
    * Generates a list containing groups of feature
    * indexes.
    *
    * @param numFeatures Number of features.
    * @param rnd A random seed
    * @return A list containing groups of feature
    *         indexes.
    */
  @Since("2.4.4")
  private def generateGroups(numFeatures: Int, rnd: Random): List[Array[Int]] = {
    if($(groupParamAsNumberOfGroups)){
      generateGroupsFromNumbers(numFeatures, rnd)
    }else{
      generateGroupsFromSizes(numFeatures, rnd)
    }
  }

  /**
    * Generates a list containing groups of feature
    * indexes given the minimum and maximum number
    * of groups to generate.
    *
    * @param numFeatures Number of features.
    * @param rnd A random seed
    * @return A list containing groups of feature
    *         indexes.
    */
  @Since("2.4.4")
  private def generateGroupsFromNumbers(numFeatures: Int, rnd: Random): List[Array[Int]] = {
    val rndFeatures = rnd.shuffle(0 to (numFeatures - 1)).toArray
    val numGroups = $(minGroup) + rnd.nextInt($(maxGroup) - $(minGroup) + 1)
    val groupSize = numFeatures / numGroups
    rndFeatures.grouped(groupSize).toList
  }

  /**
    * Generates a list containing groups of feature
    * indexes given the minimum and maximum size
    * of the groups to generate.
    *
    * @param numFeatures Number of features.
    * @param rnd A random seed
    * @return A list containing groups of feature
    *         indexes.
    */
  @Since("2.4.4")
  private def generateGroupsFromSizes(numFeatures: Int, rnd: Random): List[Array[Int]] = {
    val rndFeatures = rnd.shuffle(0 to (numFeatures - 1)).toArray
    val groupSize = $(minGroup) + rnd.nextInt($(maxGroup) - $(minGroup) + 1)
    rndFeatures.grouped(groupSize).toList
  }

  /**
   * Default copy of the classifier.
   * 
   * @param extra parameter map.
   */
  @Since("2.4.4")
  override def copy(extra: ParamMap): RotationForestClassifier = {
    defaultCopy(extra)
  }

}

/**
 * Rotation Forest Classifier model.
 * Spark port of the well-known Rotation Forest [1] algorithm.
 * 
 * [1] Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). 
 *     Rotation forest: A new classifier ensemble method. IEEE transactions 
 *     on pattern analysis and machine intelligence, 28(10), 1619-1630.
 * 
 * @author Mario Juez-Gil <mariojg@ubu.es>
 * @param uid identifier
 * @param ensemble an array containing rotation matrices and the set of
 *                 decision trees associated to them.
 * @param scalerModel the MinMaxScalerModel if normalizeData was set to true.
 * @param numClasses the number of classes.
 */
@Since("2.4.4")
class RotationForestClassificationModel private[ml] (
    @Since("2.4.4") override val uid: String,
    val ensemble: Array[(Matrix, Array[DecisionTreeClassificationModel])],
    val scalerModel: Option[MinMaxScalerModel],
    @Since("2.4.4") override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, RotationForestClassificationModel]
  with RotationForestClassifierParams with MLWritable with Serializable {

  require(ensemble.nonEmpty, "RotationForestClassificationModel requires at least" +
   " 1 ensemble element (rotation matrix + set of decision trees).")

  import RotationForestClassifier._

  /**
    * Class constructor.
    *
    * @param ensemble an array containing rotation matrices and the set of
    *                 decision trees associated to them.
    * @param scalerModel the MinMaxScalerModel if normalizeData was set to true.
    * @param numClasses the number of classes.
    * @return
    */
  private[ml] def this(
      ensemble: Array[(Matrix, Array[DecisionTreeClassificationModel])],
      scalerModel: Option[MinMaxScalerModel],
      numClasses: Int) = {
    this(Identifiable.randomUID("rotfc"), ensemble, scalerModel, numClasses)
  }

  /**
    * Transforms a dataset scaling it if needed.
    *
    * @param dataset the data to be transformed.
    * @return a DataFrame containing the transformed data.
    */
  @Since("2.4.4")
  override def transform(dataset: Dataset[_]): DataFrame = {
    scalerModel match {
      case Some(model: MinMaxScalerModel) => {
        val scaledDataset = model.transform(dataset)
          .select(SCALED_FEATURE_COL, $(labelCol))
          .withColumnRenamed(SCALED_FEATURE_COL, $(featuresCol))
        super.transform(scaledDataset)
      }
      case None => super.transform(dataset)
    }
  }

  /**
    * Predicts the output for a specific instance.
    * The output will be a vector of votes for each class.
    *
    * @param features the instance.
    * @return a vector of votes for each class.
    */
  @Since("3.0.1")
  override def predictRaw(features: Vector): Vector = {
    val votes = ensemble.map { case(rotationMatrix, trees) =>
      val rotatedFeatures = rotate(rotationMatrix, features)
      val treeVotes = Array.fill[Double](numClasses)(0.0)
      trees.view.foreach { tree =>
        val classCounts: Array[Double] = tree.rootNode.predictImpl(rotatedFeatures).impurityStats.stats
        val total = classCounts.sum
        if (total != 0) {
          var i = 0
          while (i < numClasses) {
            treeVotes(i) += classCounts(i) / total
            i += 1
          }
        }
      }
      treeVotes
    }.reduce { (rawPredictions1, rawPredictions2) => 
      (rawPredictions1, rawPredictions2).zipped.map(_ + _)
    }
    Vectors.dense(votes)
  }

  /**
    * Transforms a vector of votes for each class
    * into a vector of probabilities for each class.
    *
    * @param rawPrediction vector of votes for each class.
    * @return a vector of probabilities for each class.
    */
  @Since("2.4.4")
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in " +
          "RotationForestClassificationModel: raw2probabilityInPlace " +
          "encountered SparseVector")
    }
  }

  /**
    * Copies the model.
    *
    * @param extra extra parameters.
    * @return a copy of the model.
    */
  @Since("2.4.4")
  override def copy(extra: ParamMap): RotationForestClassificationModel = {
    copyValues(new RotationForestClassificationModel(uid, ensemble, scalerModel, numClasses), extra)
      .setParent(parent)
  }


  /**
    * For saving the model.
    * 
    * @return Model writer object.
    */
  @Since("2.4.4")
  override def write: MLWriter =
    new RotationForestClassificationModel.RotationForestClassificationModelWriter(this)

}

/**
  * Object for reading and loading Rotation Forest
  * Classification Models.
  * 
  * @author Mario Juez-Gil <mariojg@ubu.es>
  */
@Since("2.4.4")
object RotationForestClassificationModel 
  extends MLReadable[RotationForestClassificationModel] {

  /**
    * @return a model reader object.
    */
  @Since("2.4.4")
  override def read: MLReader[RotationForestClassificationModel] = {
    new RotationForestClassificationModelReader
  }

  /**
    * Loads a Rotation forest classification model given
    * the path where it is stored.
    * 
    * @return a rotation forest model.
    */
  @Since("2.4.4")
  override def load(path: String): RotationForestClassificationModel = {
    super.load(path)
  }

  /**
    * Rotation Forest Classification Model Writter
    * A MLWritter for storing the model.
    *
    * @param instance the rotation forest model.
    */
  private[RotationForestClassificationModel]
  class RotationForestClassificationModelWriter(
      instance: RotationForestClassificationModel) extends MLWriter {
    
    /**
      * Stores the model in a specific path.
      *
      * @param path where the model will be stored.
      */
    @Since("2.4.4")
    override protected def saveImpl(path: String): Unit = {
      
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numClasses" -> instance.numClasses,
        "numRotations" -> instance.getNumRotations,
        "numTrees" -> instance.getNumTrees)

      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, Some(extraMetadata))

      val ensembleData = instance.ensemble.map {
        case(rotationMatrix, trees) => {
          val rotationTrees = trees.zipWithIndex.map {
            case(tree, treeID) => {
              (DefaultParamsWriter.getMetadataToSave(tree.asInstanceOf[Params], sparkSession.sparkContext),
               EnsembleNodeData.build(tree, treeID))
            }
          }

          (rotationMatrix, rotationTrees)
        }
      }

      val scalerDataPath = new Path(path, "scaler").toString
      instance.scalerModel match {
        case Some(model: MinMaxScalerModel) => model.save(scalerDataPath)
        case None =>
      }

      val ensembleDataPath = new Path(path, "ensembleData").toString
      sparkSession.createDataFrame(ensembleData).toDF.write.parquet(ensembleDataPath)
    }

  }

  /**
    * Rotation Forest Classification Model Reader
    * A MLReader for loading the model.
    */
  private class RotationForestClassificationModelReader
    extends MLReader[RotationForestClassificationModel] {

    /** Checked against metadata when loading model */
    @Since("2.4.4")
    private val className = classOf[RotationForestClassificationModel].getName
    @Since("2.4.4")
    private val treeClassName = classOf[DecisionTreeClassificationModel].getName

    /**
      * Loads the model from a specific path.
      *
      * @param path the path where the model is stored.
      * @return the rotation forst model object.
      */
    @Since("2.4.4")
    override def load(path: String): RotationForestClassificationModel = {
      val sql = sparkSession
      import sql.implicits._

      implicit val format = DefaultFormats

      val metadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)

      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numRotations = (metadata.metadata \ "numRotations").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val impurityType: String = {
        val impurityJson: JValue = metadata.getParamValue("impurity")
        Param.jsonDecode[String](compact(render(impurityJson)))
      }

      val scalerDataPath = new Path(path, "scaler")
      val fs = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)
      val scaler = if(fs.exists(scalerDataPath)){
        Some(MinMaxScalerModel.load(scalerDataPath.toString))
      }else{
        None
      }

      val ensembleDataPath = new Path(path, "ensembleData").toString
      val ensembleData: Dataset[(Matrix, Array[(String, Seq[EnsembleNodeData])])] =
      sparkSession.read.parquet(ensembleDataPath).as[(Matrix, Array[(String, Seq[EnsembleNodeData])])]

      val ensemble = ensembleData.collect.map {
        case(rotationMatrix, treesData) => {
          val trees: Array[DecisionTreeClassificationModel] = treesData.map {
            case (jsonTreeMetadata, ensembleNodes) => {
              val treeMetadata = DefaultParamsReader.parseMetadata(jsonTreeMetadata, treeClassName)
              val nodes = ensembleNodes.map(_.nodeData)
              val root = DecisionTreeModelReadWrite.buildTreeFromNodes(nodes.toArray, impurityType)
              val tree =
                new DecisionTreeClassificationModel(treeMetadata.uid, root, numFeatures, numClasses)
              treeMetadata.getAndSetParams(tree)
              tree
            }
          }

          (rotationMatrix, trees)
        }
      }

      new RotationForestClassificationModel(metadata.uid, ensemble, scaler, numClasses)
    }
  }

}

/**
  * Parameters for Rotation Forest Classifier.
  * 
  * @author Mario Juez-Gil <mariojg@ubu.es>
  */
private[ml]
trait RotationForestClassifierParams extends RandomForestClassifierParams {
  
  /**
    * Wether minGroup and maxGroup params refer to 
    * the number of groups (true) or to its size (false).
    * 
    * In Weka is the -N param.
    */
  @Since("2.4.4")
  final val groupParamAsNumberOfGroups: BooleanParam =
    new BooleanParam(this, "groupParamAsNumberOfGroups", "Whether minGroup and " +
      "maxGroup params refer to the number of groups (true) or to its size " + 
      "(false).")

  /**
    * Minimum number of groups of sub-features when
    * groupParamAsNumberOfGrups is true. When false, 
    * refers to the minimum size of each sub-feature
    * group.
    * 
    * In Weka is the -G param.
    */
  @Since("2.4.4")
  final val minGroup: IntParam =
    new IntParam(this, "minGroup", "Minimum number of groups of " +
      "sub-features when groupParamAsNumberOfGrups is true. When false, refers to " + 
      "the minimum size of each sub-feature group.",
      ParamValidators.gtEq(1))

  /**
    * Maximum number of groups of sub-features when
    * groupParamAsNumberOfGrups is true. When false, 
    * refers to the maximum size of each sub-feature
    * group.
    * 
    * In Weka is the -H param.
    */
  @Since("2.4.4")
  final val maxGroup: IntParam =
    new IntParam(this, "maxGroup", "Maximum number of groups of " +
      "sub-features when groupParamAsNumberOfGrups is true. When false, refers to " + 
      "the maximum size of each sub-feature group.",
      ParamValidators.gtEq(1))

  /**
    * Percentage of training examples that will be
    * used for cumputing each rotation of the data.
    * 
    * In Weka is the -P param.
    */
  @Since("2.4.4")
  final val bootstrapSampleSize: DoubleParam =
    new DoubleParam(this, "bootstrapSampleSize", "Percentage of training " +
      "examples that will be used for computing each rotation of the data.",
      ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  /**
    * The number of rotations that will be performed
    * to the training data. Each rotation will be the
    * training data of a single RandomForest.
    */
  @Since("2.4.4")
  final val numRotations: IntParam =
    new IntParam(this, "numRotations", "The number of rotations that will " +
      "be performed to the training data. Each rotation will be the training " +
      "data of one RandomForest.",
      ParamValidators.gtEq(1))

  /**
    * When true, input dataset will be normalized. 
    * Data normalization could improve the performance.
    */
  @Since("2.4.4")
  final val normalizeData: BooleanParam =
    new BooleanParam(this, "normalizeData", "When true, input dataset will " +
      "be normalized. Data normalization could improve the performance. ")

  setDefault(groupParamAsNumberOfGroups -> true, minGroup -> 4, maxGroup -> 4, 
    bootstrapSampleSize -> 0.25, numRotations -> 10, normalizeData -> true, 
    numTrees -> 1)  

  /** @group getParam */
  @Since("2.4.4")
  final def getgroupParamAsNumberOfGroups: Boolean = $(groupParamAsNumberOfGroups)

  /** @group getParam */
  @Since("2.4.4")
  final def getMinGroup: Int = $(minGroup)

  /** @group getParam */
  @Since("2.4.4")
  final def getMaxGroup: Int = $(maxGroup)

  /** @group getParam */
  @Since("2.4.4")
  final def getBootstrapSampleSize: Double = $(bootstrapSampleSize)

  /** @group getParam */
  @Since("2.4.4")
  final def getNumRotations: Int = $(numRotations)

  /** @group getParam */
  @Since("2.4.4")
  final def getNormalizeData: Boolean = $(normalizeData)

}