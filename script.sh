#!/bin/bash

octave som.m
javac -classpath "lib/*" WekaMultiLayerPerceptron.java
java -classpath "lib/*:." WekaMultiLayerPerceptron
