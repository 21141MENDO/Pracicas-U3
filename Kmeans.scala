// importamos las librerias
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}

//se  Utiliza el codigo de  Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//iniciamos la sesion
val spark = SparkSession.builder().getOrCreate()
// importamos el algoritmo de clustering
import org.apache.spark.ml.clustering.KMeans

// cargamos los datos del dataset
val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")

//se pasa a modelo la data entrenada
val kmeans = new KMeans().setK(2).setSeed(1L)

val model = kmeans.fit(dataset)

//Evaluar la agrupación mediante el cálculo dentro de la suma establecida de errores al cuadrado.
val WSSE = model.computeCost(dataset)
println(s"Within set sum of Squared Errors = $WSSE")

// se muestran los resultados
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
