#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>


//Definim una funció random entre 0 i 1 per tal de fixar uns valors inicials als pesos

#define rando() ((double)rand()/((double)RAND_MAX+1))

//Definim el nombre de neurones que volem en cada capa

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 3
#define NUMOUT 1


/* run this program using the console pauser or add your own getch, system("pause") or input loop */

main (){
	
	//Definim el tipus de varibales que es necessiten
	
	//Els índexos i el nombre de neurones per capa són enters
	int    i, j, k, p, np, op, patrorandom[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
	
	//Per probar que la nostra xarxa funciona posem uns certs valors de training i de output. Simplement s'indica (0,0) o (1,1) ->0 en canvi (0,1) o (1,0) -> 1
    double Input[NUMPAT+1][NUMIN+1] = { {0, 0, 0},  {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double Target[NUMPAT+1][NUMOUT+1] = { {0, 0},  {0, 0},  {0, 1},  {0, 1},  {0, 0} };
	
	//Ara definim el tipus de variable on enregistrarem els pesos
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
	
	//Definim el tipus de variable per modificar els pesos a mesura que la xarxa apren
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
	
	//Definim els errors i altres paràmetres
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;
	
	
	//Hem d'inicialitzar els valors dels weights, farem servir random numbers
	
	//La matriu de weightIN te nº files el valor de les neurones de la primera capa més la primera que és el biaix de cada neurona, i el nº columna és la neurona cap on vaig
	
	for( j = 1 ; j <= NumHidden ; j++ ) { //Per cada neurona cap on vaig 
		for( i = 0 ; i <= NumInput ; i++ ) { //Per cada neurona d'on vinc
			DeltaWeightIH[i][j] = 0.0; //Inicialitzem la matriu de les modificacions a 0 perquè a la primera època no calculem res
			WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ; //Donem valors random a la mariu de pesos. 
		}
	}
	
	//Fem el mateix però per la matriu de pesos de la capa oculta a la capa final
	
	for (k=1; k<= NumOutput; k++){ 
		for( j = 0 ; j <= NumHidden ; j++ ) { 
			DeltaWeightHO[j][k] = 0.0; 
			WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ; 
		}
	}
	FILE*fp;
	
	fp=fopen("error_XOR.txt", "w+");
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
			
			
			//Hidden layer
			for (j=1 ; j<=NumHidden; j++){ //Aquest bucle calcula per cada neurona amagada j
				SumH[p][j]=WeightIH[0][j]; //Apunta el valor del biaix
				for (i=1 ; i<=NumInput ; i++){ //Per cada neurona d'entrada
					SumH[p][j] += Input[p][i]*WeightIH[i][j]; //Suma al biaix cada un dels pesos
				}
				Hidden[p][j]=1.0/(1.0+exp(-SumH[p][j])); //La funció sigmoide decideix si la neurona s'activa o no
			}	
			
			//Output
			for (k=1; k<=NumOutput; k++){ //Aquest bucle calcula per les neurones de la última capa que ens donen l'output. Fa el mateix procediment que abans
				SumO[p][k]=WeightHO[0][k];
				for(j=1 ; j<= NumHidden; j++){
					SumO[p][k] += Hidden[p][j]*WeightHO[j][k];
				}
				Output[p][k]=1.0/(1.0+exp(-SumO[p][k])); /*Sigmoidal outputs */
				
				//Calculem l'error que es comet amb la suma residual de quadrats
				Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]);
				
				//Decidim com variarà el pes
				DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;	
			}
			
			
			
			//Tornem enrere per modificar el hidden layer
			for (j=1 ; j<=NumHidden; j++){ 
				SumDOW[j]=0.0;
				for (k=1; k<= NumOutput; k++){
					SumDOW[j]+= WeightHO[j][k]*DeltaO[k];
				}
				DeltaH[j]=SumDOW[j]*Hidden[p][j]*(1.0-Hidden[p][j]);
			}
			for (j=1 ; j<=NumHidden; j++){
				DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j];
				WeightIH[0][j] += DeltaWeightIH[0][j];
				for (i=1 ; i<= NumInput ; i++){
					DeltaWeightIH[i][j] = eta * Input[p][j] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
					WeightIH[i][j]+=DeltaWeightIH[i][j];
				}
			}
			
			for (k=1 ; k<=NumOutput; k++){
				DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k];
				WeightHO[0][k] += DeltaWeightHO[0][k];
				for (j=1 ; j<= NumHidden ; j++){
					DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k];
					WeightHO[j][k]+=DeltaWeightHO[j][k];
				}
			}
				
		}
		if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
		fprintf(fp, "\n%f", Error) ;
        if( Error < 0.0004 ) break ;  //Que pari de computar quan ja tingui un error suficientment petit	
			
	}
	
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

    
    // Per coprovar els resultats amb el test dataset s'exteuen les matrius dels pesos
    
    FILE*fp2;
    
    fp2=fopen("matrius_XOR.txt", "w+");
    fprintf(fp2, "MATRIUIH\n") ;
    
    for (i=0 ; i<=NumInput; i++){
    	for (j=0 ; j<=NumHidden ; j++){
    		fprintf(fp2, "%f\t", WeightIH[i][j] );
		}
		fprintf(fp2,"\n");
	}
	fprintf(fp2, "\nMATRIUHO\n") ;
	for (j=0; j<=NumHidden; j++){
		for (k=0 ; k<= NumOutput; k++){
			fprintf(fp2, "%f\t", WeightHO[j][k]);
		}
		fprintf(fp2,"\n");
	}
	
	//Per probar que la nostra xarxa funciona posem uns certs valors de training i de output. Simplement s'indica (0,0) o (1,1) ->0 en canvi (0,1) o (1,0) -> 1
    double Input_2[NUMPAT+1][NUMIN+1] = { {0, 0, 0},  {0, 0.9, 0.9},  {0, 0.9, 0},  {0, 0, 0},  {0, 0, 0.9} };
    double Target_2[NUMPAT+1][NUMOUT+1] = { {0, 0},  {0, 0},  {0, 1},  {0, 0},  {0, 1} };
    
    for (p =1; p<=NumPattern; p++){
    	
		//Hidden layer
		for (j=1 ; j<=NumHidden; j++){ //Aquest bucle calcula per cada neurona amagada j
			SumH[p][j]=WeightIH[0][j]; //Apunta el valor del biaix
			for (i=1 ; i<=NumInput ; i++){ //Per cada neurona d'entrada
				SumH[p][j] += Input_2[p][i]*WeightIH[i][j]; //Suma al biaix cada un dels pesos
			}
			Hidden[p][j]=1.0/(1.0+exp(-SumH[p][j])); //La funció sigmoide decideix si la neurona s'activa o no
		}	
			
		//Output
		for (k=1; k<=NumOutput; k++){ //Aquest bucle calcula per les neurones de la última capa que ens donen l'output. Fa el mateix procediment que abans
			SumO[p][k]=WeightHO[0][k];
			for(j=1 ; j<= NumHidden; j++){
				SumO[p][k] += Hidden[p][j]*WeightHO[j][k];
			}
			Output[p][k]=1.0/(1.0+exp(-SumO[p][k])); /*Sigmoidal outputs */
				
			//Calculem l'error que es comet amb la suma residual de quadrats
			Error += 0.5 * (Target_2[p][k] - Output[p][k]) * (Target_2[p][k] - Output[p][k]);
			fprintf(stdout, "\nError%-4d = %f\n",p, Error) ;
		}
		fprintf(stdout,"\n");
	}
	
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input_2[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target_2[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;
	
    
    return 1;
}
