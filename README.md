## Breeze demo app

demo app - machine learning in Scala with Spark MLlib and the ScalaNLP Breeze library.

the Spambase data was made available for public use through the UCI Machine Learning Repository.

## Dependencies

* JVM
* SBT
* Spark 2.1.0

## Running the app

Compile the project with `sbt assembly` and run the jar with the following parameters:

```
spark-submit \
    --class BreezeDemoApp \
    --master local[2] \
    breeze-demo-assembly-0.1.jar \
    --dataset <name_of_the_dataset>.csv \
    --algorithm <name_of_the_algorithm> \
    --evaluation_file <name_of_the_evaluation_file>.txt
```
