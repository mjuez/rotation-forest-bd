# RotationForest-BD: Rotation Forest for Big Data

https://doi.org/10.1016/j.inffus.2021.03.007

This repository contains an implementation of [Rotation Forest](https://ieeexplore.ieee.org/document/1677518) [[1]](#ref_1) for Apache Spark framework.

By means of using parallel PCA provided by Spark and a novel approach for rotating the data using parallel matrix multiplications, Rotation Forest can now be used within Big Data.

RotationForest-BD is currently implemented in **Scala 2.12** for Apache **Spark 3.0.1**.

## Authors

- Mario Juez-Gil <<mariojg@ubu.es>>
- Álvar Arnaiz-González
- Juan J. Rodríguez
- Carlos López-Nozal
- César García-Osorio

**Affiliation:**\
Departamento de Ingeniería Informática\
Universidad de Burgos\
[ADMIRABLE Research Group](http://admirable-ubu.es/)

## Experiments

The experiments are available in [this repository](https://github.com/mjuez/rotation-forest-spark).

## Installation

RotationForest-BD is available on SparkPackages.

It can be installed as follows:

- **spark-shell**, **pyspark**, or **spark-submit**:
```bash
> $SPARK_HOME/bin/spark-shell --packages mjuez:rotation-forest-bd:1.0.0
```
- **sbt**:
```scala
resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "mjuez" % "rotation-forest-bd" % "1.0.0"
```
- **Maven**:
```xml
<dependencies>
  <!-- list of dependencies -->
  <dependency>
    <groupId>mjuez</groupId>
    <artifactId>rotation-forest-bd</artifactId>
    <version>1.0.0</version>
  </dependency>
</dependencies>
<repositories>
  <!-- list of other repositories -->
  <repository>
    <id>SparkPackagesRepo</id>
    <url>http://dl.bintray.com/spark-packages/maven</url>
  </repository>
</repositories>
```

## Basic Usage

RotationForest-BD is a Spark [Classifier](https://spark.apache.org/docs/latest/ml-classification-regression.html). It has a `fit` method that returns a trained classification model using an input dataset. That model has a `transform` method for classifying new instances.

RotationForest-BD may be adjusted using the following parameters:

- `groupParamsAsNumberOfGroups`: Wether minGroup and maxGroup params refer to the number of groups (true) or to its size (false). Default: `true`.
- `minGroup`: Minimum number of groups of sub-features when groupParamAsNumberOfGrups is true. When false, refers to the minimum size of each sub-feature group. Default: `4`.
- `maxGroup`: Maximum number of groups of sub-features when groupParamAsNumberOfGrups is true. When false, refers to the maximum size of each sub-feature group. Default: `4`.
- `bootstrapSampleSize`: Percentage of training examples that will be used for cumputing each rotation of the data. Default: `0.25`.
- `numRotations`: The number of rotations that will be performed to the training data. Each rotation will be the training data of a single RandomForest. Default: `10`.
- `normalizeData`: When true, input dataset will be normalized. Data normalization could improve the performance. Default: `true`.

As Rotation Forest is a tree-based ensemble, which specifically uses the Spark Random Forest implementation as base classifier, all Random Forest parameters could also be adjusted: `numTrees` `bootstrap`, `subsamplingRate`, `maxDepth`, `maxBins`, `minInstancesPerNode`, `minWeightFractionPerNode`, `minInfoGain`, `checkpointInterval`, `seed`, `maxMemoryInMB`, `leafCol`, and `cacheNodeIds`. For a detailed explanation about the use of any of those parameters, you should refer to the Spark Random Forest documentation.

The following example shows how to build and save a Rotation Forest ensemble where data is rotated 10 times and each rotation is used to train 10 trees. Thus, the ensemble size will be 100 (10x10):

```scala
import org.apache.spark.ml.classification.RotationForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline

// reading training dataset
// two columns: label, and features
val trainDS = session.read
                .format("libsvm")
                .option("inferSchema", "true")
                .load("training_dataset.libsvm")

// String Indexer configuration
val si = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("iLabel")

// Rotation Forest configuration
val rotfc = new RotationForestClassifier()
            .setLabelCol("ilabel")
            .setNumRotations(10)
            .setNumTrees(10)
            .setSeed(46)

// Building and fitting pipeline
val pipeline = new Pipeline().setStages(Array(si, rotfc))
val rotfModel = pipeline.fit(trainDS)

// Saving the model
rotfModel.write.overwrite().save("rotfmodel")
```

For loading a model and using it to make predictions, the following should be done:

```scala
import org.apache.spark.ml.PipelineModel

// reading test dataset
// one column: features
val testDS = session.read
                .format("libsvm")
                .option("inferSchema", "true")
                .load("test_dataset.libsvm")

// loading the model
val loadedRotfModel = PipelineModel.load("rotfmodel")

// making predictions
val predictDF = loadedRotfModel.transform(testDS)
```

## Contribute

Feel free to submit any pull requests 😊

## References

<a name="ref_1"></a>[1] Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). Rotation Forest: A New Classifier Ensemble Method. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(10), 1619–1630. [https://doi.org/10.1109/TPAMI.2006.211](https://doi.org/10.1109/TPAMI.2006.211)

## Aknowledgements

This work was supported through project TIN2015-67534-P (MINECO/FEDER, UE) of the *Ministerio de Economía y Competitividad* of the Spanish Government, projects BU085P17 and BU055P20 (JCyL/FEDER, UE) of the *Junta de Castilla y León* (both projects co-financed through European Union FEDER funds), and by the *Consejería de Educación* of the *Junta de Castilla y León* and the European Social Fund through a pre-doctoral grant (EDU/1100/2017). The project leading to these results has received also funding from "la Caixa" Foundation, under agreement LCF/PR/PR18/51130007. This material is based upon work supported by Google Cloud.

## License

This work is licensed under [Apache-2.0](LICENSE).

## Citation policy

Please cite this research as:

```
@ARTICLE{juezgil2021rotfbd,
title = "Rotation Forest for Big Data",
author = "Mario Juez-Gil and Álvar Arnaiz-González and Juan J. Rodríguez and Carlos López-Nozal and César García-Osorio",
journal = "Information Fusion",
year = "2021",
month = "oct",
volume = "74",
pages = "39-49",
issn = "1566-2535",
doi = "https://doi.org/10.1016/j.inffus.2021.03.007",
url = "https://www.sciencedirect.com/science/article/pii/S1566253521000634",
keywords = "Rotation Forest, Random Forest, Ensemble learning, Machine learning, Big Data, Spark",
}
```
