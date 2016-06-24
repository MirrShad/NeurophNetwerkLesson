import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class DataAppro implements LearningEventListener{


	public static int i=4;
	public static double maxError=0.001d;
	
	public static void main(String[] args) {
		File dataFile = new File("StackInfo.txt");
		try {
			FileReader fr = new FileReader(dataFile);
			BufferedReader br = new BufferedReader(fr);
			String s=br.readLine();
			DataSet trainingSetOpening=new DataSet(4, 1);
			List<Double> openingVec=new Vector<Double>();
			DataSet trainingSetEnding=new DataSet(4, 1);
			List<Double> endingVec=new Vector<Double>();
			
			while ((s = br.readLine()) != null) {
				String[] data = s.split(" ");
				
//				//process the date
//				double pointID = getPoint(data[0]);
//				
//				//process opening
				double opening = Double.parseDouble(data[1])/10000;
				openingVec.add(opening);

				//process ending
				double ending = Double.parseDouble(data[2])/10000;
				endingVec.add(ending);
				
				
			}
			
			for(int i=0;i<openingVec.size()-4;i++)
			{
				double[] inputs=new double[]{openingVec.get(i),openingVec.get(i+1),openingVec.get(i+2),openingVec.get(i+3)};
				double[] outputs=new double[]{openingVec.get(i+4)};
				trainingSetOpening.addRow(new DataSetRow(inputs,outputs));
			}
			System.out.println("predict that opening next time will be"+new DataAppro().predict(trainingSetOpening,trainingSetOpening.getRowAt(endingVec.size()-5))*10000);
			
			for(int i=0;i<endingVec.size()-4;i++)
			{
				double[] inputs=new double[]{endingVec.get(i),endingVec.get(i+1),endingVec.get(i+2),endingVec.get(i+3)};
				double[] outputs=new double[]{endingVec.get(i+4)};
				trainingSetEnding.addRow(new DataSetRow(inputs,outputs));
			}
			System.out.println("predict that ending next time will be"+new DataAppro().predict(trainingSetEnding,trainingSetEnding.getRowAt(endingVec.size()-5))*10000);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

	
	private static double getPoint(String date)
	{
		SimpleDateFormat df=new SimpleDateFormat("yyyy-MM-dd");
		Date startDate;
		long pointNum=0;
		try {
			startDate = df.parse("2014-01-01");
			Date dataDate = df.parse(date);
			pointNum =(dataDate.getTime()-startDate.getTime())/(24*60*60*1000);
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return pointNum;
	}
	
	private double predict(DataSet trainingSet,DataSetRow row)
	{
		MlPerceptron myMlPerceptron = new MlPerceptronLineOutput(TransferFunctionType.SIGMOID, 4,30,1);
		myMlPerceptron.setLearningRule(new BackPropagation());
		myMlPerceptron.getLearningRule().setLearningRate(0.2);
        myMlPerceptron.getLearningRule().setMaxError(maxError);
        myMlPerceptron.getLearningRule().addListener(this);;
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);
        System.out.println("finish learning");
        myMlPerceptron.setInput(row.getInput());
        myMlPerceptron.calculate();
        double[] output = myMlPerceptron.getOutput();
        return output[0];
	}


	@Override
	public void handleLearningEvent(LearningEvent event) {
		SupervisedLearning bp = (SupervisedLearning)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
		
	}
}
