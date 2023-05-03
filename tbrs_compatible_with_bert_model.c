/*
  Implementation of the TBRS* model (described in Oberauer & Lewandowsky, 2010)
  and inspired from the verbal theory described in Barouillet et al. (2011).
  Benoit Lemaire, Sophie Portrat, February 2013 - January 2018
  VERSION 3.2.2 : add the determ parameter to have a deterministic behavior (determ = 1)
                  add a "?" parameter to display help
                  improve the display in VERBOSE mode
  VERSION 3.2.3 : correct a bug in the processing step: process cannot be longer that free time
  VERSION INTERFERENCE : 
     Items are represented in a distributive manner, like positions. There are two kinds of
     items in the item array: memoranda (from 1 to maxMemoranda, created at the beginning) and
     distractors (from maxMemoranda+1 to maxMemoranda+maxDistractors=maxItem, created on the fly).
     Items are represented in two different forms: in a static manner (LTM) and in an
     evolutive way (WM).
     Retrieval for refreshing or recall is done by retrieving an item in WM first and then
     identifying the closest LTM representation.
     WM representations evolve because of decay (their weights move), interference (they
     tend to become a mix of current representation and distracting item) and refreshing
     (they tend to be more similar to the LTM representation)
     A parameter control the level of overlap between items and distractors (called ido)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

//CONSTANTS
#define nbUnitBlocks 9              // number of unit blocks in the position layer
#define sizeOfPositionBlocks 6      // size of unit blocks in the position layer
#define maxPosition 100              // maximum number of position
#define maxMemoranda 10             // number of items
#define maxDistractors 90           // number of distractors
#define maxItem maxMemoranda+maxDistractors
#define nbItemUnits 100             // number of units in the item layer -- for some reason this is 1 non- -1 for every 4
// 129 stops without segmentation fault, 108-128 is fault.
// below runs fine, above stops
// also the total is actually this number -1
#define distractorEncodingWeight .5 // proportion of encoding rate for distractors compared to items - .5
#define maxDisplayedUnits nbItemUnits

// COLORS
#define RED   "\x1B[1;31;47m"
#define BLU   "\x1B[1;34;47m"
#define YEL   "\x1B[1;30;43m"
#define CYN   "\x1B[1;37;46m"

#define MAG   "\x1B[35m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

// PARAMETERS 
float param_P=.3;              // Proportion of units maintained from each position to the next
float param_R=6;               // Mean memory processing rate, governing mean speed of encoding, refreshing, recall
float param_s=1;               // Standard deviation of processing rates
float param_tauE=.95;          // Criterion for encoding strength
float param_L=(float)1/9;      // Asymptot
float param_theta=.05;         // Retrieval threshold
float param_sigma=.02;         // Standard deviation of Gaussian noise added to item activations at retrieval
float param_D=.5;              // Decay rate
float param_Tr=.08;            // Mean time taken to refresh an item
float param_tauOp=.95;         // Response criterion for processing
float param_Ta=.5;             // Mean duration of attentional capture by processing steps
float param_freeTime=1;        // Free time following each processing step - 1
int param_freeTimeIncludesOpDuration=1; // free time can include or not the operation duration 
int param_refreshLastStopped=0;  // refreshing can start on the first letter (no) or on the last one when stopped (yes)  
int param_attentionalFocusSize=1; // size of the attentional focus
int nbmemo=7;                  // 7 items to memorize
float param_memoDistr=1;       // number of domain 1 items for each domain 2 item (1 means 50-50, 3 means 75-25...)
int nbop=4;                    // 4 processing operations between each item
float presentationTime=1.5;    // presentation time = 1.5s
int param_deterministic=0;     // non-deterministic behavior (seed is set to time)
float param_itemDistractorOverlap=0.4;
float param_itemDistractorNoise=1;
float param_itemItemOverlap=0.4;
int param_sameDist=0;          // Distractors are different from each other

// MODEL VARIABLES
float var_te;
float var_tr;
float var_eta;
float var_r;
float var_tauR;
float var_Rop;
float var_rop;
float var_ta;

// IMPLEMENTATION VARIABLES
int VERBOSE=0;
int PRESET=1;
int QUIET=0;
float logTauE;
float globalTime;
int nbPositionUnits;
int lastItem;
int distractorNumber;

// EMBEDDINGS LIST
float embeddingsList [maxMemoranda][nbItemUnits];

/*********/
/* ERROR */
/*********/
void  error(char *s1,char *s2) {
  printf("%s %s\n",s1,s2);
  exit(EXIT_FAILURE);
}

/*******/
/* PR2 */
/*******/
void pr2(float x) {
  // print a number less than 1 on two digits and without the leading 0
  if (x==1)
    printf(" 1 ");
  else if (x==-1)
    printf(" -1");
  else if (x==0)
    printf(" 0 ");
  else {
    int tmp=(int)(100*x);
    if (tmp<0) {
      tmp=-tmp;
      printf("-");
    }
    if (tmp<10)
      printf(".0%d",tmp);
    else
      printf(".%d",tmp);
  }
}

/********/
/* NAME */
/********/
char name(int code) {
  // return the character corresponding to an item or distractor number
  if (code<=maxMemoranda)
    if (code < 26)
      return(code+'A'-1);
    else
      return('*');
  else {
    int codeDist=code-maxMemoranda;
    if (codeDist < 10)
      return(codeDist +'1' -1);
    else
      return('#');
  }
}

/************/
/*  RANDOM  */
/************/
float randomNormal(float mean, float std) {
  // Return a random number from a normal distribution using the Box-Muller method
  float u=rand()/((double)RAND_MAX + 1);
  float v=rand()/((double)RAND_MAX + 1);
  return(mean+std*sqrt(-2*log(u))*cos(2*3.1415926535*v));
}

/*******/
/* MIN */
/******/
int min(int val1,int val2) {
  if (val1<val2)
    return(val1);
  else
    return(val2);
}

/*******/
/* MAX */
/******/
int max(int val1,int val2) {
  if (val1>val2)
    return(val1);
  else
    return(val2);
}


/**************************** **********/
/* DISPLAY ITEM POSITION ASSOCIATIONS */
/**************************************/
int displayItemPosAssociations(float itemPositionMatrix[][nbPositionUnits+1], int lastPosition) {
  // Display the first item-position associations of a given position
  int i,j;
  for(i=1;i<=lastPosition;i++) {
    printf("   %c: ",i+'A'-1);
    for(j=1;j<=3*sizeOfPositionBlocks;j++)
      printf(".%d/",(int)(1000*itemPositionMatrix[i][j])%1000);
    printf("...\n");
  }
}


/**********************/
/* DISPLAY ITEM UNITS */
/**********************/
int displayItemUnits(float itemVectorsInWM[][nbItemUnits+1],int item) {
  // Display item units - mark
  int i;
  printf("   %c: ",name(item));
  for(i=1;i<min(maxDisplayedUnits,nbItemUnits);i++)
    pr2(itemVectorsInWM[item][i]);
  printf("\n");
}


/*************************/
/* CREATE RANDOM PATTERN */    
/*************************/
void createRandomPattern(float pattern[],int min, int max) {
  // Create a random pattern from index min to index max included
  int i;
  int nb=0;
  for(i=min;i<=max;i++)
    pattern[i]=rand()/((double)RAND_MAX + 1);
}


/*********************************/
/* CREATE SIMILAR RANDOM PATTERN */    
/*********************************/
void createSimilarRandomPattern(float pattern[],float refPattern[],float std, int min, int max) {
  // Create a random pattern more or less similar to refPattern
  // Each unit has the value of the corresponding reference unit + a noise with standard deviation std,
  // from index min to index max included
  int i;
  int nb=0;
  float val;
  for(i=min;i<=max;i++) {
    if (refPattern[i]==-1)
      val=rand()/((double)RAND_MAX + 1);
    else {
      val=randomNormal(refPattern[i],std);
      if (val<0)
	val=0;
      else if (val>1)
	val=1;
    }
    pattern[i]=val;
  }
}


/*********************************/
/* CREATE OVERLAPING RANDOM PATTERN */    
/*********************************/
void createOverlapingRandomPattern(float pattern[],float refPattern[],int patternSize,float p) {
  /*This is the method to change for interfering using other memoranda*/
  // Create a new pattern which shares p% units with the reference Pattern
  // The reference pattern is filled with random values and -1. For instance:
  //     -1-1-1-1-1 x x x x x -1-1-1-1-1
  // The new pattern is filled with the same number of random values but with an
  // intersection of p%. For instance
  //     -1-1-1-1-1-1-1-1-1-1 x x x x x  if p=0
  //     -1-1-1-1-1-1-1 x x x x x -1-1-1 if p=.6
  //     -1-1-1-1-1 x x x x x -1-1-1-1-1 if p=1
  int i,c,alea,firstUsedUnit;
  float tmp;
  if (VERBOSE)
    printf("   Create a distractor sharing %d%% units with the item at previous position\n",(int)(p*100));
  //  int nbCommonUnits=patternSize/2*p;
  i=1;
  while (i<=patternSize){// && refPattern[i]==-1) { // comment out refpattern part to avoid segfault if needed
    pattern[i]=-1;
    i++;
  }
  firstUsedUnit=i;
  for(c=1;c<=(1-p)*patternSize;c++) // copy a proportion (1-p) -1s /4
    pattern[i++]=-1;
  createSimilarRandomPattern(pattern,refPattern,param_itemDistractorNoise,i,i+patternSize/4);
  // shuffle the units within the region used by the reference pattern
  for(c=firstUsedUnit+patternSize;c>firstUsedUnit;c--) { // /4
    // alea is a random number between i and c
    alea=rand()%(patternSize)+firstUsedUnit;// /4
    tmp=pattern[alea];
    pattern[alea]=pattern[c];
    pattern[c]=tmp;
  }
  for(c=i+patternSize+1;c<=patternSize;c++)// /4
    pattern[c]=-1;
}


/********************/
/* RMSE COMPUTATION */
/********************/
float rmse(float v1[],float v2[],int size) {
  // Compute the root mean square error between two vectors
  int i;
  float rmse=0,diff;
  for(i=1;i<=size;i++) {
    diff=(v1[i]-v2[i]);
    rmse+=diff*diff;
  }
  return(sqrt(rmse/size));
}

/*********/
/* DECAY */
/*********/
void decay(float itemPositionMatrix[][nbPositionUnits+1],float factor,int excludedItem) {
  // Decay item position associations
  int i,j;
  for(i=1;i<=maxItem;i++)
    if (excludedItem ==-1 || i!=excludedItem) {   // do not decay the current item
      for(j=1;j<=nbPositionUnits;j++)
        if (itemPositionMatrix[i][j]!=0)
          itemPositionMatrix[i][j]*=factor;
    }
}


/*************/
/* INTERFERE */
/*************/
void interfere(float oldItemVector[], float newItemVector[], float p) {
  // Move the old item vector features towards the new ones by a proportion p
  int j;
  for(j=1;j<=nbItemUnits;j++) {
    //    printf("%f - %f ==>",oldItemVector[j],newItemVector[j]);
    if ((oldItemVector[j] != newItemVector[j]) && ((int)oldItemVector[j]!=-1 && (int)newItemVector[j]!=-1))
      oldItemVector[j]=oldItemVector[j]*(1 -p) + newItemVector[j]*p;
    //printf("%f\n",oldItemVector[j]);
  }
}


/************/
/* RETRIEVE */
/************/
int retrieve(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],int pos,int status, float *activationMax, float *retrievalDuration,int *bestWMItem) {
  // Retrieve an item at given position (pos)
  // Status=1 ==> retrieve for refresh ; Status=0 ==> retrieve for recall
  // First, determine which WM item is best associated to the current position (bestWMItem)
  // Then, identify which LTM item is most similar to that WM item and returns it
  
  int bestItem=-1;
  char item;
  float somme;
  int i,j;
  if (status==0) {   // retrieval during recall (retrieval during refreshing has its own duration)
    float var_r=randomNormal(param_R,param_s);
    if (var_r<.1)
      var_r=.1;
    var_tr=logTauE/var_r;   // cf Eq. 3a in Oberauer & Lewandowsky (2010)
    if (var_tr > presentationTime)
      var_tr=presentationTime;
  }
  *retrievalDuration=var_tr;
  *activationMax=-99999;
  if (VERBOSE) printf("[%.2fs] ",globalTime);
  float activationValues[maxMemoranda+distractorNumber+1];
  for(item=1;item<=maxMemoranda+distractorNumber;item++) {  // activation values of each item are computed
    somme=0;

    // compute activation from position units
    for(i=1;i<=nbPositionUnits;i++)
      if (positionVectors[pos][i]!=0)
	somme+=positionVectors[pos][i] * itemPositionMatrix[item][i];

    // add noise
    somme+=randomNormal(0,1) * max(param_sigma,.0001);

    if (VERBOSE)
      activationValues[item]=somme;
    
    // get maximum value
    if (somme > *activationMax) {
      *activationMax=somme;
      *bestWMItem=item;
    }
  }

  // display activation values
  if (VERBOSE) {
    int switchToDistractors=0;
    for(item=1;item<=maxMemoranda+distractorNumber;item++) {  // activation values of each item are compu
      if (item > maxMemoranda && switchToDistractors==0) {
	printf("... ");
	switchToDistractors=1;
      }
      if (item == *bestWMItem)
	printf(RED);
      if (item <= 26 && item<=maxMemoranda) {  // memoranda
	printf("%c:",item+'A'-1);
	pr2(activationValues[item]);
      }
      else if (item > maxMemoranda) {// distractors
	printf("%d:",item-maxMemoranda);
	pr2(activationValues[item]);
      }
      if (item == *bestWMItem)
	printf(RESET);
      printf(" ");
    }
  }
  
  if (VERBOSE)
    printf("\n");

  int retrievedItem;
  if (*activationMax < param_theta) {
    retrievedItem=0;
    if (VERBOSE) printf("No memoranda are above the theta threshold.\n");
  }
  else {
    // Comparison between the retrieved item and the stable LTM item representations
    float minRmse=99999;
    float aRmse;
    for(item=1;item<=maxMemoranda+distractorNumber;item++) {
      aRmse=rmse(itemVectorsInWM[*bestWMItem],itemVectorsInLTM[item],nbItemUnits);
      if (aRmse<minRmse) {
	minRmse=aRmse;
	retrievedItem=item;
      }
    }
    if (VERBOSE) {
      printf("   ");
      printf(CYN "Pos%d: %c (%.2f) is the closest LTM item to the best WM item (%c) (RMSE=%.4f)" RESET,pos,name(retrievedItem),*activationMax,name(*bestWMItem),minRmse);
      printf("\n");
    }
  }
  return(retrievedItem);
}


/**********/
/* ENCODE */
/**********/
float encode(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],int initialEncoding,int currentItem,int bestWMItem,int position,float timeLeft,int strengthDivisor,float duration, int distractor) {
  // Encode the current symbol in given position
  // Return the encodingDuration
  // distractor = 1 if it is the encoding of a distractor
  int i,j,retrievedItem,currentItemSymbol;
  float encodingDuration, factor,var_r,retrievalDuration,activationMax;

  if (distractor)
    currentItemSymbol=currentItem-maxMemoranda+'0';
  else
    currentItemSymbol=currentItem+'A'-1;

  // compute the value of r which is drawn from the mean R
  var_r=randomNormal(param_R,param_s);
  if (var_r<.1)
    var_r=.1;

  if (initialEncoding) {
    if (distractor) {
      encodingDuration=0;  // it is the processing time for distractors
    }
    else {
    // copy LTM representation into WM at initial encoding
      for (j=1;j<=nbItemUnits;j++)
	itemVectorsInWM[currentItem][j]=itemVectorsInLTM[currentItem][j];    
      var_te=logTauE/var_r;
      if (var_te > presentationTime)
	var_te=presentationTime;
      encodingDuration=var_te;

      if (VERBOSE) printf("[%.2fs]   (%d) Encoding duration of %c = %1.3f\n",globalTime,distractor,currentItemSymbol,encodingDuration);

    var_eta=1-exp(-var_r*encodingDuration);
    }

    // new item interfere with all other items
    // NOT USED FOR THE MOMENT. KEEP THE CODE THOUGH.
    //    for(i=1;i<=maxMemoranda+distractorNumber;i++)
    //      interfere(itemVectorsInWM[i],itemVectorsInWM[currentItem],.4);
    //    if (VERBOSE) {
    //      printf("   All items are altered by the new encoded item.\n");
    //      for(i=1;i<=lastItem;i++)
    //	if (i!=currentItem)
    // displayItemUnits(itemVectorsInWM,i);
    // }

  } // reencoding
  else {   // reencoding during refreshing
    if (duration == -1) {  // duration is not given and has to be computed
      var_tr=-log(1-var_tauR)/var_r;             // cf Eq. 3a in Oberauer & Lewandowsky (2010)
      if (var_tr > timeLeft)
	var_tr=timeLeft;
      encodingDuration=var_tr;
    }
    else 
      encodingDuration=duration;
    var_eta=1-exp(-var_r*encodingDuration);  // vaut param_tauR la plupart du temps sauf quand var_te > presentationTime
    var_eta/=strengthDivisor;  // if more than one item is considered at the same time, divise the strength accordingly
  }

  globalTime+=encodingDuration;

  // Decay during encoding of memoranda
  if (!distractor && duration == -1)  // do not decay if duration is given, which means it has been done before
    decay(itemPositionMatrix,exp(-param_D * encodingDuration),currentItem);

  // Encoding of a distractor	    
  // first, retrieve the WM memoranda at current position, then alter it with distractor
  int tmp;
  if (distractor) {
    retrievedItem=retrieve(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,position,1,&activationMax,&retrievalDuration,&tmp);

    // create distractor pattern
    if (param_sameDist == 1) { // distractors are all the same
      if (distractorNumber == 0) { // first distractor of the trial
	distractorNumber=1;
	createOverlapingRandomPattern(itemVectorsInLTM[maxMemoranda+distractorNumber],itemVectorsInWM[retrievedItem],nbItemUnits,param_itemDistractorOverlap);
      }
    }
    else // distractors are different from each other
      createOverlapingRandomPattern(itemVectorsInLTM[maxMemoranda+distractorNumber],itemVectorsInWM[retrievedItem],nbItemUnits,param_itemDistractorOverlap);

    // copy LTM representation into WM
    for (j=1;j<=nbItemUnits;j++)
      itemVectorsInWM[maxMemoranda+distractorNumber][j]=itemVectorsInLTM[maxMemoranda+distractorNumber][j];    

    if (VERBOSE) {
      displayItemUnits(itemVectorsInWM,maxMemoranda+distractorNumber);
      displayItemUnits(itemVectorsInWM,lastItem);
    }
    
    interfere(itemVectorsInWM[retrievedItem],itemVectorsInWM[maxMemoranda+distractorNumber],distractorEncodingWeight);
    
    if (VERBOSE) {
      printf("   ");
      printf(YEL "%c is retrieved at position %d and altered by the distractor" RESET,name(retrievedItem),position);
      printf("\n");
      displayItemUnits(itemVectorsInWM,retrievedItem);
    }

    var_eta*=distractorEncodingWeight;   // distractor are weakly encoded
  }

  // Create or update association links between items and positions
  for(j=1;j<=nbPositionUnits;j++)                
    if (positionVectors[position][j] != 0)
      itemPositionMatrix[currentItem][j]+=(param_L-itemPositionMatrix[currentItem][j])*var_eta;
 
  // Update item representation: get closer to the LTM representation
  if (!initialEncoding && !distractor) { //  refreshing
    if (VERBOSE)
      printf("   Move WM item %c closer to LTM item %c\n", name(bestWMItem),name(currentItem));
    interfere(itemVectorsInWM[bestWMItem],itemVectorsInLTM[currentItem],.5);
    if (VERBOSE)
      displayItemUnits(itemVectorsInWM,bestWMItem);
  }
  
  return(encodingDuration);
}

/*******************************/
/* COMPARE STIMULUS AND RECALL */
/*******************************/
float compareStimAndRecalled(char recalled[],int lastPosition,float resSerialPositionData[]) {
  // Compare the recalled sequence with the stimulus ("ABC...") and return the proportion correct
  int i;
  int sommeOrdre=0;
  for (i=0;i<=lastPosition-1;i++)
    if (recalled[i] == 'A'+i) {
      sommeOrdre++;
      resSerialPositionData[i+1]++;
    }
  return((float)sommeOrdre/lastPosition);
}

/***********/
/* REFRESH */
/***********/
int refresh(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],float timeAvailable,int lastPosition) {
  // Refresh an item
  int currentPosition=1;
  int cpi,afsi;
  int bestLTMItem,bestWMItem;
  float reencodingDuration;
  float activationMax,retrievalDuration;
  
  while (timeAvailable > 0) {
    afsi = min(param_attentionalFocusSize,lastPosition);
    cpi = currentPosition;
    reencodingDuration=-1;  // reencoding duration is not known yet. Use the previous one afterwards

    while (afsi > 0) {
      bestLTMItem=retrieve(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,cpi,1,&activationMax,&retrievalDuration,&bestWMItem);
      
      if (VERBOSE) {
	printf("   ");
	printf(CYN "It is refreshed." RESET);
	printf("\n");
      }
      reencodingDuration=encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,0,bestLTMItem,bestWMItem,cpi,timeAvailable,min(param_attentionalFocusSize,lastPosition),reencodingDuration,0);

      if (VERBOSE)
	printf("   %c is reencoded in %1.3f ms\n",name(bestLTMItem),reencodingDuration);
      cpi++;
      if (cpi > lastPosition)
	cpi=1;
      afsi--;
    }
    currentPosition=cpi;
    timeAvailable-=reencodingDuration;
    if (timeAvailable < 0)
      timeAvailable=0;
  }
}


/**************/
/* PROCESSING */
/**************/
float processing(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],int lastPosition) {

  int i,j;
  float factor;
  var_Rop=-log(1-param_tauOp)/param_Ta;    
  var_rop=randomNormal(var_Rop,param_s);  //draw a random value r >=.1
  if (var_rop<.1)
    var_rop=.1;
  var_ta=-log(1-param_tauOp)/var_rop;             // cf Eq. 3a in Oberauer & Lewandowsky (2010)
  if ((var_ta > param_freeTime) && (param_freeTimeIncludesOpDuration==1)) {
    if (VERBOSE) printf("   Process stopped. Planned to last %1.3f ms but no free time left.\n",var_ta);
    var_ta=param_freeTime;
  }
  if (VERBOSE) printf("[%.2fs]   Processing duration=%1.3f\n",globalTime,var_ta);

  // create distractor pattern
  //  createOverlapingRandomPattern(itemVectorsInWM[maxMemoranda+distractorNumber],itemVectorsInWM[lastItem],nbItemUnits,param_itemDistractorOverlap);
  
  //if (VERBOSE) {
  //displayItemUnits(itemVectorsInWM,maxMemoranda+distractorNumber);
  //displayItemUnits(itemVectorsInWM,lastItem);
  //}
  
  // Encode distractor
  float encodingDuration=encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,1,maxMemoranda+distractorNumber,-1,lastPosition,9999,1,var_ta,1);

  globalTime += var_ta;

  // all items decay
  decay(itemPositionMatrix,exp(-param_D*var_ta),-1);

  return(var_ta);
}

/*
// CRUDE AUTOPILOT MODE CODE
float processing(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],int lastPosition) {

  int i,j;
  float factor;
  var_Rop=-log(1-param_tauOp)/param_Ta;    
  var_rop=randomNormal(var_Rop,param_s);  //draw a random value r >=.1
  if (var_rop<.1)
    var_rop=.1;
  var_ta=-log(1-param_tauOp)/var_rop;             // cf Eq. 3a in Oberauer & Lewandowsky (2010)
  if ((var_ta > param_freeTime) && (param_freeTimeIncludesOpDuration==1)) {
    if (VERBOSE) printf("   Process stopped. Planned to last %1.3f ms but no free time left.\n",var_ta);
    var_ta=param_freeTime;
  }
  if (VERBOSE) printf("[%.2fs]   Processing duration=%1.3f\n",globalTime,var_ta);

  // Encode distractor at all positions
  float encodingDuration = 0;
  for (int pos = 1; pos <= nbPositionUnits; pos++) {
    encodingDuration += encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,1,maxMemoranda+distractorNumber,pos,lastPosition,9999,1,var_ta,1);
  }

  globalTime += var_ta;

  // all items decay
  decay(itemPositionMatrix,exp(-param_D*var_ta),-1);

  return(var_ta);
}

*/

/**********/
/* RECALL */
/**********/
void recall(int positionVectors[][nbPositionUnits+1],float itemVectorsInWM[][nbItemUnits+1],float itemVectorsInLTM[][nbItemUnits+1],float itemPositionMatrix[][nbPositionUnits+1],float itemStrength[],int lastPosition, char recalled[maxPosition+1]) {
  int i,j,position,codeItem;
  int bestLTMItem;
  int bestWMItem;
  float retrievalDuration,activationMax,factor;

  for(position=1;position<=lastPosition;position++) {  // items are recalled according to their position
    bestLTMItem=retrieve(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,position,0,&activationMax,&retrievalDuration,&bestWMItem);
    if (retrievalDuration > 5)
      retrievalDuration=5;

    // decay during recall
    decay(itemPositionMatrix,exp(-param_D*retrievalDuration),-1);
    //    factor=exp(-param_D*retrievalDuration);
    //    for(i=1;i<=maxItem;i++) {
    //      itemStrength[i]*=factor;
    //  for(j=1;j<=nbPositionUnits;j++) 
    //	if (itemPositionMatrix[i][j]!=0) 
    //	  itemPositionMatrix[i][j]*=factor;
    //}

    if (activationMax > param_theta) {
      if ((bestLTMItem>=1) && (bestLTMItem<=26))
	codeItem=bestLTMItem+'A'-1;
      else 
	codeItem='*';
      // Response suppression
      //      for (i=1;i<=nbItemUnits;i++)
      //if (positionVectors[position][i] != 0)
      for(j=1;j<=nbPositionUnits;j++)
	if (itemVectorsInWM[bestLTMItem][j] != 0)
	  itemPositionMatrix[bestLTMItem][j]-=param_L*activationMax;
    }
    else
      codeItem='.';
    if (VERBOSE) {
      printf("   ");
      printf(RED "position %d: %c is retrieved" RESET,position,codeItem);
      printf("\n");
    }
    recalled[position-1]=codeItem;
    globalTime+=retrievalDuration;
  }
  recalled[lastPosition]='\0';
}

	

/*************************************/
/* GENERATE POSITION REPRESENTATIONS */
/*************************************/
void generatePositionRepresentations(int positionVectors[][nbPositionUnits+1]) {
  int i,j,p,randomPosition,oldPosition;
  
  for(i=1;i<=nbPositionUnits;i++)
    positionVectors[1][i]=0;
  for(i=0;i<=nbUnitBlocks-1;i++) {
    randomPosition=1+rand()%sizeOfPositionBlocks;
    positionVectors[1][i*sizeOfPositionBlocks+randomPosition]=1;
  }
  
  for(p=2;p<=maxPosition;p++) {
    for(i=0;i<=nbUnitBlocks-1;i++) {
      if (rand()%100>param_P*100) {  //if the block has to be changed
	for(j=i*sizeOfPositionBlocks+1;j<=(i+1)*sizeOfPositionBlocks;j++) {
	  //	  if (positionVectors[p-1][j]==1) 
	  //  oldPosition=j; //previous position of the 1 is memorized
	  positionVectors[p][j]=0;  // its units are set to 0
	}
	randomPosition=1+rand()%sizeOfPositionBlocks;  // and a new "1" is added
	positionVectors[p][i*sizeOfPositionBlocks+randomPosition]=1;
      }
      else {  // the block has to remain unchanged
	for(j=i*sizeOfPositionBlocks+1;j<=(i+1)*sizeOfPositionBlocks;j++) {
	  positionVectors[p][j] = positionVectors[p-1][j]; // it is the same as the previous one
	}
      }
    }
  }
  
  // DISPLAY
  if (VERBOSE) {
    printf("POSITION UNITS\n");
    for(p=1;p<=maxPosition;p++) {
      printf("   [%2d] ",p);
      for(i=1;i<=nbPositionUnits;i++) 
	printf("%d",positionVectors[p][i]);
      printf("\n");
    }
  }
}

/*********************************/
/* GENERATE ITEM REPRESENTATIONS */
/*********************************/
void generateItemRepresentations(float itemVectors[][nbItemUnits+1], float d1, float d2) {
  // generate distributed representations for all items, d1% in domain 1, d2% in domain 2 - mark
   // Items in domain 1 use unit indexes from 1 to nbItemUnits/4
   // Items in domain 2 use unit indexes from nbItemUnits/4+1 to nbItemUnits/2
  int i,j,p;
  for(i=1;i<=maxMemoranda;i++) { // memoranda
    for(j=1;j<=nbItemUnits;j++)
      itemVectors[i][j]=-1; 
    //   if (i%2==1) // Item in domain 1
    if (1) // Item in domain 1
      if(!PRESET)
        createRandomPattern(itemVectors[i],1,nbItemUnits); // basically only in domain 1 -- used to be nbitemunits/4
      else{
        for(int jj=1;jj<=nbItemUnits;jj++){
          itemVectors[i][jj]=embeddingsList[i-1][jj-1];
          //printf("%d", embeddingsList[i-1][jj-1]);
        }
      }
    //else // Item in domain 2
      //createRandomPattern(itemVectors[i],nbItemUnits/4+1,nbItemUnits/2);	
      
  }
  
  for(i=maxMemoranda+1;i<=maxItem;i++) { // distractors. They will be instantiated on the fly
    for (j=1;j<=nbItemUnits;j++)
      itemVectors[i][j]=-1;  // meaning that the item is not characterized along those dimensions
  }

  // DISPLAY
  if (VERBOSE) {
    printf("ITEM UNITS (first however many units)\n");
    for(p=1;p<=min(26,maxMemoranda);p++) // why's it min(26, maxMemoranda) here
      displayItemUnits(itemVectors,p);
  }
}


/********/
/* MAIN */
/********/
int main(int argc,char* argv[]) {
  printf("running");

  float span;
   span = 0;
   int nbmemo_in;
   
   int k;
   for(k=1;k<=nbmemo;k++){
    nbmemo_in = k;
    printf("%i",nbmemo);

  nbPositionUnits=nbUnitBlocks*sizeOfPositionBlocks;  // number of units in the position layer

  float itemPositionMatrix[maxItem+1][nbPositionUnits+1];
  float itemStrength[maxItem+1];   // WM representations of item strengths
  int nbSimulations=500;
  int positionVectors[maxPosition+1][nbPositionUnits+1];
  float itemVectorsInWM[maxItem+1][nbItemUnits+1];    // WM representations of items
  float itemVectorsInLTM[maxItem+1][nbItemUnits+1];   // LTM representations of items
  int lastPosition;
  int i,j;

  // READ EMBEDDINGS LIST FROM FILE


  #define ROWS maxMemoranda
  #define COLS nbItemUnits

  
  FILE *fp;
  int ik, jk;
  float **arr;

  // Open the file for reading
  fp = fopen("pca_embeddings_c.txt", "r");
  if (fp == NULL) {
      printf("Error opening file\n");
      return 1;
  }

  // Allocate memory for the array
  arr = (float **) malloc(ROWS * sizeof(float *));
  for (ik = 0; ik < ROWS; ik++) {
      arr[ik] = (float *) malloc(COLS * sizeof(float));
  }

  // Read data from the file
  for (ik = 0; ik < ROWS; ik++) {
      for (jk = 0; jk < COLS; jk++) {
          fscanf(fp, "%f,", &arr[ik][jk]);
      }
  }
  // Print the array
  for (ik = 0; ik < ROWS; ik++) {
      for (jk = 0; jk < COLS; jk++) {
          printf("%f ", arr[ik][jk]);
          embeddingsList[ik][jk] = arr[ik][jk];
      }
      printf("\n");
  }

  // Free memory
  for (ik = 0; ik < ROWS; ik++) {
      free(arr[ik]);
  }
  free(arr);


  // Close the file
  fclose(fp);
      
      
      
  

  
  char *syntax="Syntaxe:\n\
  ?                  this message\n\
  -v                 verbose (disabled by default)\n\
  -q(uiet)           only display statistics, no recall data\n\
  nbmemo <value>     number of items (default=7)\n\
  memoDistr <value>  percentage of memo in domain 1 (number of memo in domain 2 is nbmemo-this value)\n\
  nbop <value>       number of operations (default=4)\n\
  R <value>          Mean memory processing rate (default=6)\n\
  P <value>          Proportion of units maintained from each position to the next (default=.3)\n\
  s <value>          Standard deviation of processing rates (default=1)\n\
  theta <value>      Retrieval threshold (default=.05)\n\
  sigma <value>      Standard deviation of Gaussian noise added to item activations at retrieval (default=.02)\n\
  D <value>          Decay rate (default=.5)\n\
  Tr <value>         Mean time taken to refresh an item (default=.08)\n\
  Ta <value>         Mean duration of attentional capture by processing steps (default=.5)\n\
  freeTime <value>   Free time following each processing step (default=1)\n\
  ftiod <0 or 1>     Free time can include (1) or not (0) the operation duration (default=1)\n\
  determ <0 or 1>    Model can be deterministic (1) or not (0) (default=0)\n\
  sameDist <0 or 1>  Indicates if distractors are identical (0) or all different (1)\n\
  idn <value>        Standard deviation of the noise used to create distractor wrt memorand\n\
  ido <value>        Item-distractor overlap\n"; 

  // Analyze command line
  i=1;
  while (i<argc) {
    if (!strcmp(argv[i],"?")) {error(syntax,"");i++;}
    else if (!strcmp(argv[i],"-v")) {VERBOSE=1;i++;}
    else if (!strcmp(argv[i],"-q")) {QUIET=1;i++;}
    else if (!strcmp(argv[i],"-n")) {nbSimulations=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"nbmemo")) {nbmemo=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"memoDistr")) {param_memoDistr=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"nbop")) {nbop=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"R")) {param_R=atof(argv[i+1]);i+=2;} 
    else if (!strcmp(argv[i],"P")) {param_P=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"s")) {param_s=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"D")) {param_D=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"theta")) {param_theta=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"sigma")) {param_sigma=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"Tr")) {param_Tr=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"Ta")) {param_Ta=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"freeTime")) {param_freeTime=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"ftiod")) {param_freeTimeIncludesOpDuration=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"determ")) {param_deterministic=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"sameDist")) {param_sameDist=atoi(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"idn")) {param_itemDistractorNoise=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"iio")) {param_itemItemOverlap=atof(argv[i+1]);i+=2;}
    else if (!strcmp(argv[i],"ido")) {param_itemDistractorOverlap=atof(argv[i+1]);i+=2;}
    else 
      error("Unknown parameter:",argv[i]);
  }
  
  if (nbop>16)
    error("Cannot handle more than 16 operations.","");

  if (param_itemDistractorOverlap<0 || param_itemDistractorOverlap>1)
    error("Item distractor overlap should be between 0 and 1.","");

  // initialize random generator
  
  if (param_deterministic)
    srand(param_deterministic);
  else 
    srand(time(0));    
  
  int cptReplic=1;
  float resPropCorrect=0;

  //CREATE STIMULUS
  char stimuli[nbmemo * (nbop+1)+1];
  int cpt=0;
  for(i=0;i<nbmemo;i++) {
    stimuli[cpt++]=65+i;  // letters for memoranda
    for(j=1;j<=nbop;j++) {
      stimuli[cpt++]=j+'0';  //numbers for distractors
    }
  }
  stimuli[cpt]='#';        //recall
    
  //INITIALIZE SERIAL POSITION DATA
  float resSerialPositionData[maxPosition+1];
  for (i=1;i<=nbmemo;i++)
    resSerialPositionData[i]=0;

  int symbol, idxstimulus;
  float processingDuration;
  char recalled[maxPosition+1];

  logTauE = -log(1-param_tauE);
  var_tauR = 1-exp(-param_R * param_Tr);
  
  while (cptReplic <= nbSimulations) {  

    // Generate position representations
    generatePositionRepresentations(positionVectors);

    // Generate item representations (100% in domain 1, 0% in domain 2)
    //generateItemRepresentations(itemVectorsInLTM,param_memoDistr,1-param_memoDistr);
    generateItemRepresentations(itemVectorsInLTM,1,0);
    // this above line operates on a 101x101 matrix which is rly weird

    
    
    //INITIALISATION OF THE ITEM x POSITION MATRIX
    for(i=1;i<=maxItem;i++)
      for(j=1;j<=nbPositionUnits;j++)
	      itemPositionMatrix[i][j]=0;

    //INITIALISATION OF ITEM STRENGTHS
    for(i=1;i<=nbItemUnits;i++)
      itemStrength[i]=0;
    
    lastPosition=0;
    idxstimulus=0;
    distractorNumber=0;

    // MAIN LOOP
    while(1) {
      //printf(sizeof stimuli);
      symbol=stimuli[idxstimulus++];

      // Processing a memoranda
      if (symbol>='A' && symbol <='Z') {
        //printf("\n processing memo" + symbol + "m");
        if (VERBOSE) {
          printf(RED "\n   MEMORIZING %c   \n" RESET,symbol);
          printf("\n");
        }
        lastPosition++;
        lastItem=symbol-'A'+1;
        float encodingDuration=encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,1,symbol-'A'+1,-1,lastPosition,presentationTime,1,-1,0);
        if (VERBOSE) displayItemPosAssociations(itemPositionMatrix,lastPosition);
        if (encodingDuration<0) 
          error("There should be no error in initial encoding...","");
        refresh(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,presentationTime-encodingDuration,lastPosition);
	      if (VERBOSE) displayItemPosAssociations(itemPositionMatrix,lastPosition);
      }
      
      // Processing a distractor
      else if (symbol>='0' && symbol <= '9'+7) {   // we can have up to 16 operations in a row - mark
        //printf("\n processing distractor" + symbol + "d");
        if (VERBOSE) {
          printf(RED "\n   PROCESSING ");
          if (symbol > '9')
            printf("1%c",symbol-10);
          else 
            printf("%c",symbol);
          printf("   \n" RESET);
          printf("\n");
        }
        if (distractorNumber>maxDistractors)
          error("number of distractors is higher than what is allowed in the program. Increase maxDistractors constant","");
        if (param_sameDist == 0)  // distractors are different from each other
          distractorNumber++;  
        processingDuration=processing(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,lastPosition);

        float timeLeft;
        if (param_freeTimeIncludesOpDuration) 
          timeLeft=param_freeTime-processingDuration;
        else 
          timeLeft=param_freeTime;
        if (VERBOSE) displayItemPosAssociations(itemPositionMatrix,lastPosition);
        
        // autopilot
        // Encode distractor
        //float somethingsomething=encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,1,maxMemoranda+distractorNumber,-1,lastPosition,9999,1,timeLeft,1);
        //decay(itemPositionMatrix,exp(-param_D*timeLeft),-1);
        
        // right stuff
        refresh(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,timeLeft,lastPosition);
        //float somethingsomething=encode(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,1,maxMemoranda+distractorNumber,-1,lastPosition,9999,1,timeLeft,1);
        
        // all items decay
        if (VERBOSE) displayItemPosAssociations(itemPositionMatrix,lastPosition);


      }

      // Recall
      else if (symbol == '#') {
	if (VERBOSE) {
	  printf(RED "\n   RECALL   \n" RESET);
	  printf("\n");
	}
	recall(positionVectors,itemVectorsInWM,itemVectorsInLTM,itemPositionMatrix,itemStrength,lastPosition,recalled);
	if (!QUIET) // print recall data
	  fprintf(stderr,"#%4d: Recalled = %s\n",cptReplic,recalled);
	resPropCorrect+=compareStimAndRecalled(recalled,lastPosition,resSerialPositionData);
	break;
      }
      else 
	error("Unknown symbol in stimulus","");
    }
    
    cptReplic++;
  }

  // DISPLAY RESULTS
  // Headings
  printf("NBSimulations NbMemo NbOp ProportionCorrect P R s tauE L theta sigma D Tr tauOp Ta freeTime ftIncludesOp refreshLastStopped attentionalFocusSize ");

  // Data
  printf("\n"); 
  printf("%d %d %d %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %d %d %d ",nbSimulations,nbmemo_in,nbop,resPropCorrect/nbSimulations,param_P,param_R,param_s,param_tauE,param_L,param_theta,param_sigma,param_D,param_Tr,param_tauOp,param_Ta,param_freeTime,param_freeTimeIncludesOpDuration,param_refreshLastStopped,param_attentionalFocusSize);
  printf("\n");

  printf("Span \n"); 
  for(i=1;i<=nbmemo_in;i++)
    printf("Pos%d ",i);
  printf("\n"); 
  
  for(i=1;i<=nbmemo_in;i++)
    printf("%1.4f ",resSerialPositionData[i]/nbSimulations);
  printf("\n");
  span = span + resPropCorrect/nbSimulations;
  printf("%1.4f",span);
   }
}
