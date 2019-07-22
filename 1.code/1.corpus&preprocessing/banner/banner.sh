CP=${CP}:lib/banner.jar
CP=${CP}:lib/dragontool.jar
CP=${CP}:lib/heptag.jar
CP=${CP}:lib/commons-collections-3.2.1.jar
CP=${CP}:lib/commons-configuration-1.6.jar
CP=${CP}:lib/commons-lang-2.4.jar
CP=${CP}:lib/mallet-deps.jar
CP=${CP}:lib/mallet.jar
CP=${CP}:lib/commons-logging-1.1.1.jar
FUNCTION=$1
CONFIG=$2
P1=$3
P2=$4
java -cp ${CP} banner.eval.BANNER $FUNCTION $CONFIG $P1 $P2

