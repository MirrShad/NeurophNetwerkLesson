import java.util.Arrays;
import java.util.Scanner;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.util.TransferFunctionType;

public class AdalineDemo {
	public final static int CHAR_WIDTH = 5;
    public final static int CHAR_HEIGHT = 7;
    public static String[][] DIGITS = { 
            { 
              " 000 ", 
              "0   0", 
              "0   0", 
              "0   0", 
              "0   0",
              "0   0", 
              " 000 "
             },
            { 
                  "  0  ", 
                  " 00  ", 
                  "0 0  ", 
                  "  0  ", 
                  "  0  ",
                  "  0  ", 
                  "  0  " 
                 

            }, {
                
                  " 000 ", 
                  "0   0", 
                  "    0", 
                  "   0 ", 
                  "  0  ",
                  " 0   ", 
                  "00000"         
            }, {
                
                  " 000 ", 
                  "0   0", 
                  "    0", 
                  " 000 ", 
                  "    0",
                  "0   0", 
                  " 000 "     

            }, {
                
                  "   0 ", 
                  "  00 ", 
                  " 0 0 ", 
                  "0  0 ", 
                  "00000",
                  "   0 ", 
                  "   0 "     
                
                

            }, {
                  "00000", 
                  "0    ", 
                  "0    ", 
                  "0000 ", 
                  "    0",
                  "0   0", 
                  " 000 "     

            }, {
                  " 000 ", 
                  "0   0", 
                  "0    ", 
                  "0000 ", 
                  "0   0",
                  "0   0", 
                  " 000 " 

            }, {
                  "00000", 
                  "    0", 
                  "    0", 
                  "   0 ", 
                  "  0  ",
                  " 0   ", 
                  "0    " 
                
                

            }, {
                
                  " 000 ", 
                  "0   0", 
                  "0   0", 
                  " 000 ", 
                  "0   0",
                  "0   0", 
                  " 000 " 

            }, {
                
                  " 000 ", 
                  "0   0", 
                  "0   0", 
                  " 0000", 
                  "    0",
                  "0   0", 
                  " 000 " 
                
                

            }
    };
	public static void main(String[] args) {
		Adaline ada = new Adaline(CHAR_WIDTH * CHAR_HEIGHT,DIGITS.length,0.01d,TransferFunctionType.LINEAR);
        DataSet ds = new DataSet(CHAR_WIDTH * CHAR_HEIGHT, DIGITS.length);
        for (int i = 0; i < DIGITS.length; i++) {
            //一个数字符号就是一个训练的数据，第0个数字的的期望输出为0，第一个数字的期望输出为1等等。
            ds.addRow(createTrainDataRow(DIGITS[i],i));            
        }        
        ada.learn(ds);
        /*for (int i = 0; i < DIGITS.length; i++)
        {
        	ada.setInput(image2data(DIGITS[i]));
            ada.calculate();
            printDIGITS(DIGITS[i]);        
            System.out.println(maxIndex(ada.getOutput()));
            System.out.println(Arrays.toString(ada.getOutput()));
            System.out.println();            
        }*/
        Scanner s = new Scanner(System.in);
        String[] temp = new String[CHAR_HEIGHT];
        for(int i=0;i<CHAR_HEIGHT;i++)
        {
        	temp[i]=s.nextLine();
        }
        ada.setInput(image2data(temp));
        ada.calculate();
        System.out.println(maxIndex(ada.getOutput()));
        System.out.println(Arrays.toString(ada.getOutput()));
        System.out.println();    
	}
	
	private static int maxIndex(double[] output) {
		double maxData=output[0];
        int maxIndex=0;
        for (int i = 0; i < output.length; i++) {
            if(maxData<output[i]){
                maxData=output[i];
                maxIndex=i;            
            }
        }        
        return maxIndex;
	}

	private static void printDIGITS(String[] image) {
		for (int i = 0; i < image.length; i++) {
            System.out.println(image[i]);            
        }
        System.out.println("\n");
		
	}

	private static double[] image2data(String[] image) {
		double[] input=new double[CHAR_WIDTH*CHAR_HEIGHT];
        //行的长度，即为字符的长度，为整个字体的高度
        for (int row = 0; row < CHAR_HEIGHT; row++) {
            //有多少个列
            for (int col = 0; col < CHAR_WIDTH; col++) {
                int index=(row*CHAR_WIDTH)+col;
                char ch=image[row].charAt(col);
                input[index]=ch=='0'?1:-1;                
            }
        }
        
        return input;
	}

	private static DataSetRow createTrainDataRow(String[] image, int idealValue) {
		double[] output=new double[DIGITS.length];
        for (int i = 0; i < output.length; i++) {
            output[i]=-1;
        }        
        double[] input=image2data(image);        
        output[idealValue]=1;
        DataSetRow dsr=new DataSetRow(input,output);                
        return dsr;
	}

	public void handleLearningEvent(LearningEvent event) {
        // TODO Auto-generated method stub
        
    }
}
