#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>


//Definim una funció random entre 0 i 1 per tal de fixar uns valors inicials als pesos

#define rando() ((double)rand()/((double)RAND_MAX+1))

//Definim el nombre de neurones que volem en cada capa

#define NUMPAT 150 //TRAINING
#define NUMIN  2
#define NUMHID1 4
#define NUMHID2 4
#define NUMOUT 1 
#define NUMPAT2 50 //TEST


/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main (){
	
	//Definim el tipus de varibales que es necessiten
	
	//Els índexos i el nombre de neurones per capa són enters
	int    i, j, k, p, np, op, patrorandom[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden1 = NUMHID1, NumHidden2=NUMHID2, NumOutput = NUMOUT;
	
	//Per probar que la nostra xarxa funciona posem uns certs valors de training i de output. Simplement s'indica (0,0) o (1,1) ->0 en canvi (0,1) o (1,0) -> 1
	
	double Input[NUMPAT+1][NUMIN+1];
    double Target[NUMPAT+1][NUMOUT+1];
    
    double INP[NUMPAT+1][NUMIN+NUMOUT+1];
	
	//Obrim el fitxer
	int r,c;
	FILE * file;
	file=fopen("nombres_train.txt","r");
	for (c=0; c<4; c++){
		INP[0][c]=0.0;
	}

	for (r=1; r<151;r++){
		for (c=1; c<=3; c++){
			if(!fscanf(file, "%lf", &INP[r][c]))
				break;
		}
	}
	fclose(file);
	
	for (r=0;r<NUMPAT+1;r++){
		for (c=0; c<NUMIN+1;c++){
			Input[r][c]=INP[r][c];
		}
		
	}
	
	for (r=0;r<NUMPAT+1;r++){
		for (c=3; c<NUMIN+NUMOUT+1;c++){
			Target[r][1]=INP[r][c];
		}
		
	}

	
	//for (r=0; r<NUMPAT+1; r++){
	//	for (c=0; c<NUMIN+1; c++){
	//		printf("%lf     ", Input[r][c]);
	//	}
	//		printf("\n");
	//}
	
	//for (r=0;r<NUMPAT+1;r++){
	//	for (c=0; c<NUMOUT+1;c++){
	//		printf("%lf     ", Target[r][c]);
	//	}
	//		printf("\n");
	//}
	
	//Ara definim el tipus de variable on enregistrarem els pesos
    double SumH1[NUMPAT+1][NUMHID1+1], WeightIH1[NUMIN+1][NUMHID1+1], Hidden1[NUMPAT+1][NUMHID1+1];
    double SumH2[NUMPAT+1][NUMHID2+1], WeightH1H2[NUMHID1+1][NUMHID2+1], Hidden2[NUMPAT+1][NUMHID2+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightH2O[NUMHID2+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
	
	//Definim el tipus de variable per modificar els pesos a mesura que la xarxa apren
    double DeltaO[NUMOUT+1], SumDOW2[NUMHID2+1], SumDOW1[NUMHID1+1], DeltaH1[NUMHID1+1], DeltaH2[NUMHID2+1];
    double DeltaWeightIH1[NUMIN+1][NUMHID1+1], DeltaWeightH1H2[NUMHID1+1][NUMHID2+1], DeltaWeightH2O[NUMHID2+1][NUMOUT+1];
	
	//Definim els errors i altres paràmetres
    double Error, eta = 0.1, alpha = 0.3, smallwt = 0.5;
	
	
	//Hem d'inicialitzar els valors dels weights, farem servir random numbers
	
	//La matriu de weightIN te nº files el valor de les neurones de la primera capa més la primera que és el biaix de cada neurona, i el nº columna és la neurona cap on vaig
	
	for( j = 1 ; j <= NumHidden1 ; j++ ) { //Per cada neurona cap on vaig 
		for( i = 0 ; i <= NumInput ; i++ ) { //Per cada neurona d'on vinc
			DeltaWeightIH1[i][j] = 0.0; //Inicialitzem la matriu de les modificacions a 0 perquè a la primera època no calculem res
			WeightIH1[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ; //Donem valors random a la mariu de pesos. 
		}
	}
	
	for( j = 1 ; j <= NumHidden2 ; j++ ) { //Per cada neurona cap on vaig 
		for( i = 0 ; i <= NumHidden1 ; i++ ) { //Per cada neurona d'on vinc
			DeltaWeightH1H2[i][j] = 0.0; //Inicialitzem la matriu de les modificacions a 0 perquè a la primera època no calculem res
			WeightH1H2[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ; //Donem valors random a la mariu de pesos. 
		}
	}
	
	//Fem el mateix però per la matriu de pesos de la capa oculta a la capa final
	
	for (k=1; k<= NumOutput; k++){ 
		for( j = 0 ; j <= NumHidden2 ; j++ ) { 
			DeltaWeightH2O[j][k] = 0.0; 
			WeightH2O[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ; 
		}
	}
	
	FILE*fp3;
	
	fp3=fopen("error_punts_2c.txt", "w+");
	
	//Posem un nombre suficient d'iteracions per tal de que vagi buscant els paràmetres adequats dels pesos
	for ( epoch=0; epoch < 100000 ; epoch++){	
		//Caldrà que a cada iteració ordeni de manera diferent les dades que tenim perquè sinó sempre farà el mateix procés
		for (p=1 ; p<=NumPattern ; p++){
			patrorandom[p]=p;
		}
		
		for (p=1 ; p<=NumPattern ; p++){
			np= p + rando() * (NumPattern+1-p);
			op= patrorandom[p] ; patrorandom[p] = patrorandom[np] ; patrorandom[np]=op;
		}
		Error=0.0;
		
		//Necessitem que faci els càlculs per tots els patrons amb els corresponents diferents ordres
		
		for (np =1; np<=NumPattern; np++){
			p=patrorandom[np];
			
			
			//Hidden layer 1
			for (j=1 ; j<=NumHidden1; j++){ //Aquest bucle calcula per cada neurona amagada j
				SumH1[p][j]=WeightIH1[0][j]; //Apunta el valor del biaix
				for (i=1 ; i<=NumInput ; i++){ //Per cada neurona d'entrada
					SumH1[p][j] += Input[p][i]*WeightIH1[i][j]; //Suma al biaix cada un dels pesos
				}
				Hidden1[p][j]=1.0/(1.0+exp(-SumH1[p][j])); //La funció sigmoide decideix si la neurona s'activa o no
			}	
			
			//Hidden layer 2
			for (j=1 ; j<=NumHidden2; j++){ //Aquest bucle calcula per cada neurona amagada j
				SumH2[p][j]=WeightH1H2[0][j]; //Apunta el valor del biaix
				for (i=1 ; i<=NumHidden1 ; i++){ //Per cada neurona d'entrada
					SumH2[p][j] += Hidden2[p][i]*WeightH1H2[i][j]; //Suma al biaix cada un dels pesos
				}
				Hidden2[p][j]=1.0/(1.0+exp(-SumH2[p][j])); //La funció sigmoide decideix si la neurona s'activa o no
			}	
			
			//Output
			for (k=1; k<=NumOutput; k++){ //Aquest bucle calcula per les neurones de la última capa que ens donen l'output. Fa el mateix procediment que abans
				SumO[p][k]=WeightH2O[0][k];
				for(j=1 ; j<= NumHidden2; j++){
					SumO[p][k] += Hidden2[p][j]*WeightH2O[j][k];
				}
				Output[p][k]=1.0/(1.0+exp(-SumO[p][k])); /*Sigmoidal outputs */
				
				//Calculem l'error que es comet amb la suma residual de quadrats
				Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]);
				
				//Decidim com variarà el pes
				DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;	
			}
			
			
			
			//Tornem enrere per modificar el hidden layer 2
			for (j=1 ; j<=NumHidden2; j++){ 
				SumDOW2[j]=0.0;
				for (k=1; k<= NumOutput; k++){
					SumDOW2[j]+= WeightH2O[j][k]*DeltaO[k];
				}
				DeltaH2[j]=SumDOW2[j]*Hidden2[p][j]*(1.0-Hidden2[p][j]);
			}
			
			//Tornem enrere per modificar el hidden layer 1
			for (j=1 ; j<=NumHidden1; j++){ 
				SumDOW1[j]=0.0;
				for (k=1; k<= NumHidden2; k++){
					SumDOW1[j]+= WeightH1H2[j][k]*DeltaH2[k];
				}
				DeltaH1[j]=SumDOW1[j]*Hidden2[p][j]*(1.0-Hidden2[p][j]);
			}
			
			//Tornem a l'inici
			for (j=1 ; j<=NumHidden1; j++){
				DeltaWeightIH1[0][j] = eta * DeltaH1[j] + alpha * DeltaWeightIH1[0][j];
				WeightIH1[0][j] += DeltaWeightIH1[0][j];
				for (i=1 ; i<= NumInput ; i++){
					DeltaWeightIH1[i][j] = eta * Input[p][j] * DeltaH1[j] + alpha * DeltaWeightIH1[i][j];
					WeightIH1[i][j]+=DeltaWeightIH1[i][j];
				}
			}
			
			for (j=1 ; j<=NumHidden2; j++){
				DeltaWeightH1H2[0][j] = eta * DeltaH2[j] + alpha * DeltaWeightH1H2[0][j];
				WeightH1H2[0][j] += DeltaWeightH1H2[0][j];
				for (i=1 ; i<= NumHidden1 ; i++){
					DeltaWeightH1H2[i][j] = eta * Hidden1[p][j] * DeltaH2[j] + alpha * DeltaWeightH1H2[i][j];
					WeightH1H2[i][j]+=DeltaWeightH1H2[i][j];
				}
			}
			
			for (k=1 ; k<=NumOutput; k++){
				DeltaWeightH2O[0][k] = eta * DeltaO[k] + alpha * DeltaWeightH2O[0][k];
				WeightH2O[0][k] += DeltaWeightH2O[0][k];
				for (j=1 ; j<= NumHidden2 ; j++){
					DeltaWeightH2O[j][k] = eta * Hidden2[p][j] * DeltaO[k] + alpha * DeltaWeightH2O[j][k];
					WeightH2O[j][k]+=DeltaWeightH2O[j][k];
				}
			}
				
		}
		if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
		fprintf(fp3, "\n%f", Error) ;
        if( Error < 0.004 ) break ;  //Que pari de computar quan ja tingui un error suficientment petit	
			
	}
	
	fclose(fp3);
	
	//Per escriure els resultats en un txt
	
	FILE*fp;
	
	fp=fopen("training_results_2c.txt", "w+");
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(fp, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(fp, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(fp, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(fp, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(fp, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;
	fclose(fp);
	
	
	//Per visualitzar els resultats de la xarxa neuronal
	 
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;  
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;
    
	// S'exteuen les matrius dels pesos
    
    FILE*fp2;
    
    fp2=fopen("matrius_punts.txt", "w+");
    fprintf(fp2, "MATRIUIH1\n") ;
    
    for (i=0 ; i<=NumInput; i++){
    	for (j=0 ; j<=NumHidden1 ; j++){
    		fprintf(fp2, "%f\t", WeightIH1[i][j] );
		}
		fprintf(fp2,"\n");
	}
	
	fprintf(fp2, "MATRIUH1H2\n") ;
	for (i=0 ; i<=NumInput; i++){
    	for (j=0 ; j<=NumHidden2 ; j++){
    		fprintf(fp2, "%f\t", WeightH1H2[i][j] );
		}
		fprintf(fp2,"\n");
	}
	
	fprintf(fp2, "\nMATRIUH2O\n") ;
	for (j=0; j<=NumHidden2; j++){
		for (k=0 ; k<= NumOutput; k++){
			fprintf(fp2, "%f\t", WeightH2O[j][k]);
		}
		fprintf(fp2,"\n");
	}
	
	fclose(fp2);
	
    return 1 ;
	
}
	
