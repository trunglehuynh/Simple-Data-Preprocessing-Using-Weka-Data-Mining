package TrungHuynh;
// Data Mining
// Homework 1
// Name: Trung Huynh

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.core.*;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.matrix.Matrix;

public class Main {

	public static void main(String[] str) throws IOException {

		print("++++++++++++  Part One   ++++++++++++\n");
		partOne();
		
		print("\n++++++++++++  Part Two   ++++++++++++\n");
		partTwo();
	}

	// part two
	private static void partTwo() {
		// setup data set
		int sizeData = 10;
		Attribute x = new Attribute("X", Attribute.NUMERIC);
		Attribute y = new Attribute("Y", Attribute.NUMERIC);
		ArrayList<Attribute> attribute = new ArrayList<>();
		attribute.add(x);
		attribute.add(y);
		Instances data = new Instances("Gaussian", attribute, sizeData);
		Random ran = new Random();
		double[] arrayX, arrayY;
		double averageX = 0.0, averageY = 0.0;

		for (int i = 0; i < sizeData; i++) {
			Instance inst = new DenseInstance(2);
			inst.setValue(x, ran.nextGaussian());
			inst.setValue(y, ran.nextGaussian());
			data.add(inst);
		}
		arrayX = data.attributeToDoubleArray(0);
		arrayY = data.attributeToDoubleArray(1);

		for (int i = 0; i < arrayX.length; i++) {

			averageX = averageX + arrayX[i];
			averageY = averageY + arrayY[i];

		}

		averageX = averageX / arrayX.length;
		averageY = averageY / arrayY.length;

		Matrix deviationMatrix = new Matrix(sizeData, 2);

		for (int i = 0; i < arrayX.length; i++) {

			deviationMatrix.set(i, 0, arrayX[i] - averageX);
			deviationMatrix.set(i, 1, arrayY[i] - averageY);
		}

		Matrix coverianceMatr = deviationMatrix.transpose().times(deviationMatrix);

		print("the covariance matrix Σ");
		print(coverianceMatr);

		print("Size of data is: " + sizeData);
		print("\n++++++++the nearest neighbor using Euclidean distance++++++++\n");
		int[] neighborArrayEuclidean = new int[data.size()];
		for (int i = 0; i < data.size(); i++) {
			Instance a = data.get(i);
			double MinDis = Double.MAX_VALUE;
			int minZ = 0;
			for (int z = 0; z < data.size(); z++) {
				if (i == z)
					continue;

				Instance b = data.get(z);
				double dis = EuclideanDistance(a, b);
				if (dis < MinDis) {
					MinDis = dis;
					minZ = z;
				}

			}

			neighborArrayEuclidean[i] = minZ;
			print("the nearest neighbor of point at index: " + i + " is point at index: " + minZ);
			print("the distance is: " + MinDis);
			print("");
		}

		print("\n++++++++The nearest neighbor using Mahalanobis distance++++++++\n");

		int[] neighborArrayMahalanobis = new int[data.size()];
		for (int i = 0; i < data.size(); i++) {
			Instance a = data.get(i);
			double MinDis = Double.MAX_VALUE;
			int minZ = 0;
			for (int z = 0; z < data.size(); z++) {
				if (i == z)
					continue;

				Instance b = data.get(z);
				double dis = MahalanobisDistance(a, b, coverianceMatr);
				if (dis < MinDis) {
					MinDis = dis;
					minZ = z;
				}

			}
			neighborArrayMahalanobis[i] = minZ;
			print("the nearest neighbor of point at index: " + i + " is point at index: " + minZ);
			print("the distance is: " + MinDis);
			print("");
		}

		double count = 0;

		for (int i = 0; i < neighborArrayEuclidean.length; i++) {
			if (neighborArrayEuclidean[i] == neighborArrayMahalanobis[i]) {
				++count;
			}
		}

		print("The Number of the nearest neighbors are same: " + count);
		print("the consistency ratio: " + (count / neighborArrayEuclidean.length));
	}

	// find Manalanobis distance
	private static double MahalanobisDistance(Instance a, Instance b, Matrix covariance) {
		double dis = 0;
		double[] val = { a.value(0) - b.value(0), a.value(1) - b.value(1) };

		Matrix ab = new Matrix(val, 1);

		dis = ab.times(covariance.inverse()).times(ab.transpose()).trace();
		return dis;

	}

	// find EuclideanDistance
	private static double EuclideanDistance(Instance a, Instance b) {

		double dis = Math.sqrt(Math.pow((a.value(0) - b.value(0)), 2) + Math.pow((a.value(1) - b.value(1)), 2));

		return dis;

	}

	// part one
	private static void partOne() throws IOException {
		// reading file from data folder
		BufferedReader reader = new BufferedReader(new FileReader("data/weather.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);

		Map<String, Integer> map = new HashMap<String, Integer>();
		Map<String, Double> MaxAttributes = new HashMap<String, Double>();
		Map<String, Double> MinAttributes = new HashMap<String, Double>();

		print("the number of possible attribute values: ");
		for (int i = 0; i < data.numAttributes(); i++) {

			if (data.attribute(i).isNominal()) {

				print(data.attribute(i).name() + " : " + data.attribute(i).numValues());

				// prepare question for 1b
				for (int y = 0; y < data.attribute(i).numValues(); y++) {
					String value = data.attribute(i).value(y);

					// if (!map.containsKey(value)) {
					map.put(value, 0);
					// }
				}
			}

			if (data.attribute(i).isNumeric()) {
				MaxAttributes.put(data.attribute(i).name(), Double.MIN_VALUE);
				MinAttributes.put(data.attribute(i).name(), Double.MAX_VALUE);
			}

		}

		// (b) the number of instances for each attribute value
		for (int i = 0; i < data.size(); i++) {

			Instance ins = data.get(i);
			// count instances for each attribute value
			for (int y = 0; y < ins.numAttributes(); y++) {
				if (ins.attribute(y).isNominal()) {

					String valueName = ins.stringValue(y);

					map.put(valueName, map.get(valueName) + 1);

				}

				if (ins.attribute(y).isNumeric()) {

					double val = ins.value(y);
					// find max
					if (MaxAttributes.get(ins.attribute(y).name()) < val) {
						MaxAttributes.put(ins.attribute(y).name(), val);
					}
					// find min
					if (MinAttributes.get(ins.attribute(y).name()) > val) {
						MinAttributes.put(ins.attribute(y).name(), val);
					}

				}

			}

		}
		print("\nthe number of instances for each attribute value:");

		for (int i = 0; i < data.numAttributes(); i++) {

			if (data.attribute(i).isNominal()) {

				System.out.print(data.attribute(i).name() + ":");
				for (int z = 0; z < data.attribute(i).numValues(); z++) {
					String name = data.attribute(i).value(z);
					System.out.print(" " + name + " " + map.get(name) + ",");
				}
				print("");
			}

		}
		// 2. For a numeric attribute
		// (a) output the maximum and minimum value

		print("\noutput the maximum value");
		printMapDouble(MaxAttributes);

		print("\noutput the minimum value");
		printMapDouble(MinAttributes);

		// (b) split the range from minimum to maximum into two equal-length
		// intervals (say lower and upper),
		// and output the number of instances lower and upper intervals,
		// respectively.

		for (Map.Entry<String, Double> entry : MaxAttributes.entrySet()) {

			double average = (entry.getValue() + MinAttributes.get(entry.getKey())) / 2;
			int upper = 0, lower = 0;
			Attribute attr = data.attribute(entry.getKey());
			for (int i = 0; i < data.size(); i++) {
				Instance ins = data.get(i);

				if (ins.value(attr) > average) {
					++upper;
				} else {
					++lower;
				}
			}
			print("");
			print(attr.name());
			print("Average: " + average);
			print("number of instances lower: " + lower);
			print("number of instances upper: " + upper);

		}
	}

	// print all values in a map
	private static void printMapDouble(Map<String, Double> ma) {
		for (Map.Entry<String, Double> entry : ma.entrySet()) {
			String key = entry.getKey().toString();
			Double value = entry.getValue();
			print(key + " : " + value);
		}
	}

	// simple print out results
	private static void print(Object mess) {
		System.out.println(mess.toString());
	}

}
