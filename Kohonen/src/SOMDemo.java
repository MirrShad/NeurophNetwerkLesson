import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Vector;

import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Kohonen;

public class SOMDemo {

	public static void main(String[] args) {
		final int dataNum = 44;
		String[] dataKey = new String[dataNum];
		Vector<Vector> dataMatrix = new Vector<Vector>();
		Vector<Vector> data = new Vector<Vector>();
		double[][] dataArray = new double[dataNum][dataNum+23];
		
		//data reading
		File file = new File("CityData.txt");
		BufferedReader reader = null;
		try{
			reader = new BufferedReader(new FileReader(file));
			String tempString =null;
			int keyCount =0;
			while((tempString = reader.readLine())!=null)
				{
					String[] as = tempString.split(" ");
					Vector<String> dataRow = new Vector<String>();
					for(String temp:as)
					{
						dataRow.addElement(temp);
					}
					dataKey[keyCount]=as[0];
					dataMatrix.addElement(dataRow);
					keyCount++;
				}
			reader.close();
		}catch(Exception e)
		{
			e.printStackTrace();
		}finally{
			if(reader!=null){
				try{
					reader.close();
				}catch(Exception e)
				{
					
				}
			}
		}
		
		//data process
		for(int i =0;i<dataNum;i++)//the ith row
		{
			Vector<Integer> dataRow = new Vector<Integer>();
			for(int j=0;j<dataNum;j++)
			{
				if(j==i)
				{
					dataRow.addElement(1);
				}
				else
				{
					dataRow.addElement(0);
				}
			}
			
			for(int k=1;k<6;k++)//the kth character
			{
				if(k==1)
				{
					float GDP =Float.parseFloat((String)dataMatrix.get(i).get(k));
					for(int m=0;m<5;m++)
					{
						dataRow.addElement(0);
					}
					if(GDP<10000)
					{
						dataRow.set(dataNum+0, 1);
					}else if (GDP<20000)
					{
						dataRow.set(dataNum+1, 1);
					}else if (GDP<30000)
					{
						dataRow.set(dataNum+2, 1);
					}else if (GDP<40000)
					{
						dataRow.set(dataNum+3, 1);
					}else
					{
						dataRow.set(dataNum+4, 1);
					}
				}
				
				if(k==2)
				{
					float industrialAmount = Float.parseFloat((String)dataMatrix.get(i).get(k));
					for(int m=0;m<5;m++)
					{
						dataRow.addElement(0);
					}
					if(industrialAmount<100)
					{
						dataRow.set(dataNum+5, 1);
					}else if (industrialAmount<300)
					{
						dataRow.set(dataNum+6, 1);
					}else if (industrialAmount<800)
					{
						dataRow.set(dataNum+7, 1);
					}else if (industrialAmount<1000)
					{
						dataRow.set(dataNum+8, 1);
					}else
					{
						dataRow.set(dataNum+9, 1);
					}
				}
				
				if(k==3)
				{
					float goodAmount = Float.parseFloat((String)dataMatrix.get(i).get(k));
					for(int m=0;m<5;m++)
					{
						dataRow.addElement(0);
					}
					if(goodAmount<50)
					{
						dataRow.set(dataNum+10, 1);
					}else if (goodAmount<100)
					{
						dataRow.set(dataNum+11, 1);
					}else if (goodAmount<300)
					{
						dataRow.set(dataNum+12, 1);
					}else if (goodAmount<500)
					{
						dataRow.set(dataNum+13, 1);
					}else
					{
						dataRow.set(dataNum+14, 1);
					}
				}
				

				if(k==4)
				{
					float retailAmount = Float.parseFloat((String)dataMatrix.get(i).get(k));
					for(int m=0;m<4;m++)
					{
						dataRow.addElement(0);
					}
					if(retailAmount<200)
					{
						dataRow.set(dataNum+15, 1);
					}else if (retailAmount<300)
					{
						dataRow.set(dataNum+16, 1);
					}else if (retailAmount<600)
					{
						dataRow.set(dataNum+17, 1);
					}else
					{
						dataRow.set(dataNum+18, 1);
					}
				}
				
				if(k==5)
				{
					float localAmount = Float.parseFloat((String)dataMatrix.get(i).get(k));
					for(int m=0;m<4;m++)
					{
						dataRow.addElement(0);
					}
					if(localAmount<5000)
					{
						dataRow.set(dataNum+19, 1);
					}else if (localAmount<10000)
					{
						dataRow.set(dataNum+20, 1);
					}else if (localAmount<20000)
					{
						dataRow.set(dataNum+21, 1);
					}else
					{
						dataRow.set(dataNum+22, 1);
					}
				}
			}
	
			data.addElement(dataRow);
		}
		//data vector to array
		for(int i=0;i<dataNum;i++)
		{
			for(int j=0;j<(dataNum+23);j++)
			{
				int temp =(int) data.get(i).get(j);
				dataArray[i][j]=temp;
			}
		}
		
		//analysize
		ResultFrame frame = new ResultFrame();
		Kohonen som=new Kohonen(dataNum+23,400);
		DataSet ds = new DataSet(dataNum+23);
		for (double[] row : dataArray) {
			ds.addRow(new DataSetRow(row));
		}
		som.learn(ds);
		for (int i=0;i<dataArray.length;i++) {
			som.setInput(dataArray[i]);
			som.calculate();
			int winnerIndex=getWinnerIndex(som);
			int x=getRowFromIndex(winnerIndex);
			int y=getColFromIndex(winnerIndex);
			System.out.println(dataKey[i]+" "+x+" "+y );
			frame.addElementString(new ResultFrame.ElementString(dataKey[i], x, y));
		}
		frame.showMe();
	}

	private static int getWinnerIndex(Kohonen neuralNetwork) {
		Neuron winner = new Neuron();
		double minOutput = 100;
		int winnerIndex=-1;
		Neuron[] neurons=neuralNetwork.getLayerAt(1).getNeurons();
		for (int i=0;i<neurons.length;i++) {
			double out = neurons[i].getOutput();
			if (out < minOutput) {
				minOutput = out;
				winnerIndex = i;
			} // if
		} // while
		return winnerIndex;
	}
	
	private static int getRowFromIndex(int index){
		return index/20+1;
	}
	private static int getColFromIndex(int index){
		return index%20+1;
	}
}
