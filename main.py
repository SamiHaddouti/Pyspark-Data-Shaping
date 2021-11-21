"""Import all necessary packages"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import  (
    split as _split,
    mean as _mean,
    sum as _sum,
    col as _col,
    when,
    isnan,
    count,
)
from pyspark.ml.feature import Imputer

spark = SparkSession\
        .builder\
        .appName('Exercise1')\
        .getOrCreate()

# Load data as data frame and keep column names + data types
# from original data set with head = True and inferSchema = True
# Add escape option to ignore " in the csv text
df_pyspark = spark.read\
    .option("escape","\"")\
    .csv('titanic.csv', header = True, inferSchema = True)

"""
# First look at the data, columns and data types

print((df_pyspark.count(), len(df_pyspark.columns)))
df_pyspark.show()
print(df_pyspark.printSchema())
print(df_pyspark.columns)
print(df_pyspark.dtypes)
df_pyspark.describe().show()

# Find out how many missing values each column has
df_pyspark.select([count(when(isnan(c) | _col(c).isNull(), c))\
                .alias(c) for c in df_pyspark.columns]).show()

print((df_pyspark.count(), len(df_pyspark.columns)))      # Check shape before shaping
"""

# Drop not needed columns and replace/rename missing values
df_pyspark = df_pyspark.drop('Ticket')   # Not necessary for my intentions
df_pyspark = df_pyspark.na.drop(how = 'any', thresh = 10)   # Drop rows with at least two NA values
df_pyspark = df_pyspark.na.drop(how = 'any', subset = ['Cabin']) # Drop rows with missing values
                                                                 # in cabin column
# Replace missing age values by median
imputer = Imputer(
    inputCols = ['Age'],
    outputCols = ['{}_imputed'.format(c) for c in ['Age']]
 ).setStrategy('median')
df_pyspark = imputer.fit(df_pyspark).transform(df_pyspark)
df_pyspark = df_pyspark.drop('Age')                          # Drop old Age column

df_pyspark = df_pyspark.na.fill('unknown')   # Replace NA values in Embarked with description 'unknown'

# Generate and add new interesting columns
family_column = when((_col('SibSp') > 0) | (_col('Parch') > 0), True).otherwise(False)
df_pyspark = df_pyspark.withColumn('Family', family_column)\
                        .withColumn('FamilySize', df_pyspark['SibSp'] + df_pyspark['Parch'])

# Rename columns for better understanding
df_pyspark = df_pyspark.withColumnRenamed('Family', 'isFamily')     # Because only boolean
df_pyspark = df_pyspark.withColumnRenamed('SibSp', 'NumberOfSibOrSp')\
                       .withColumnRenamed('Parch', 'NumberOfParOrChild')\
                       .withColumnRenamed('Pclass', 'Class')
df_pyspark = df_pyspark.withColumnRenamed('Age_imputed', 'Age')     # Change name back to Age

# Filter for only fare above null and isFamily = true rows, then drop is family
df_pyspark = df_pyspark.filter('Fare > 0')

# Split Name column into title, firstname and lastname
df_pyspark = df_pyspark.withColumn('Lastname', _split(_col('Name'), ', ').getItem(0))\
                       .withColumn('TitleFirstname', _split(_col('Name'), ', ').getItem(1))
df_pyspark = df_pyspark.withColumn('Title', _split(_col('TitleFirstname'), '. ').getItem(0))\
                        .withColumn('Firstname', _split(_col('TitleFirstname'), '\ ').getItem(1))

# Rearrange, select columns and sort by age
df_pyspark = df_pyspark.select('PassengerId', 'Title', 'Firstname', 'Lastname', 'Sex', 'Age', \
                               'Survived', 'Class', 'Embarked', 'Fare', 'Cabin', 'isFamily', \
                               'FamilySize', 'NumberOfSibOrSp', 'NumberOfParOrChild').sort('Age')

# Find interesting values
family_fares = df_pyspark.groupBy('Lastname').agg(_sum('Fare'))
death_rate_by_families = df_pyspark.groupBy('Lastname').agg(_mean('Survived'))
death_rate_by_gender = df_pyspark.groupBy('Sex').agg(_mean('Survived'))
death_rate_by_title = df_pyspark.groupBy('Title').agg(_mean('Survived'))
death_rate_by_class = df_pyspark.groupBy('Class').agg(_mean('Survived'))
death_rate_by_cabin = df_pyspark.groupBy('Cabin').agg(_mean('Survived'))
mean_survivor_age = df_pyspark.groupBy('Survived').agg(_mean('Age'))

#family_fares.show()
#death_rate_by_families.show()
#death_rate_by_gender.show()
#death_rate_by_title.show()
#death_rate_by_class.show()
#death_rate_by_cabin.show()
#mean_survivor_age.show()
#df_pyspark.show()

#print((df_pyspark.count(), len(df_pyspark.columns)))      # Check shape after shaping

# Save new data frame as csv

# Save each partition individually -> slicing bigger csv file for better scaling
df_pyspark.write.csv('titanic_family_refined.csv')
# Bundle partitions
df_pyspark.repartition(1).write.format("com.databricks.spark.csv")\
                .option("header", "true").save("titanic_family_refined2.csv")
"""
# Alternatives
df_pyspark.coalesce(1).write.format("com.databricks.spark.csv")\
                .option("header", "true").save("titanic_family_refined3.csv")
df_pyspark.toPandas().to_csv('titanic_family_refined4.csv')   # Save as single csv file
"""

# Stopping Spark Session and freeing potential cluster resources
spark.stop()
