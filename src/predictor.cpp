#include <stdio.h>
#include <math.h>
#include "predictor.h"


const char *studentName = "Rohit Ramaprasad";
const char *studentID = "";
const char *email = "";


// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
                         "Tournament", "Custom"};

// define number of bits required for indexing the BHT here.
int ghistoryBits = 14; // Number of bits used for Global History
int bpType;            // Branch Prediction Type
int verbose;

// gshare
uint8_t *bht_gshare;
uint64_t ghistory;


// gshare functions
void init_gshare()
{
  int bht_entries = 1 << ghistoryBits;
  bht_gshare = (uint8_t *)malloc(bht_entries * sizeof(uint8_t));
  int i = 0;
  for (i = 0; i < bht_entries; i++)
  {
    bht_gshare[i] = WN;
  }
  ghistory = 0;
}

uint8_t gshare_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  switch (bht_gshare[index])
  {
  case WN:
    return NOTTAKEN;
  case SN:
    return NOTTAKEN;
  case WT:
    return TAKEN;
  case ST:
    return TAKEN;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    return NOTTAKEN;
  }
}

void train_gshare(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;

  // Update state of entry in bht based on outcome
  switch (bht_gshare[index])
  {
  case WN:
    bht_gshare[index] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_gshare[index] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  // Update history register
  ghistory = ((ghistory << 1) | outcome);
}

void cleanup_gshare()
{
  free(bht_gshare);
}

// **************************************
// **************************************

uint32_t hashPC(uint32_t pc, uint64_t ghr, uint32_t num_bits)
{
  return ((pc & ((1<<(num_bits+2))-1))>>2) ^ (ghr & ((1<<num_bits)-1));
}

typedef struct PBP {
  uint64_t ghr;
  uint64_t choice_table_size;
  uint64_t chooser_counter_size;
  uint64_t chooser_counter_half_mark;
  uint8_t* chooser;

  uint32_t *pht;
  uint32_t pht_table_size;
  uint32_t pht_bits;
  uint8_t *lbht;
  uint32_t lbht_counter_size;
  uint32_t lbht_counter_half_mark;

  uint64_t ghr_redundant_1_num;
  uint64_t ghr_redundant_1_to_use;

  uint64_t ghr_redundant_2_num;
  uint64_t ghr_redundant_2_to_use;

  uint64_t ghr_redundant_3_num;
  uint64_t ghr_redundant_3_to_use;

} PBP;

typedef struct Perceptron {
  uint32_t size_budget;
  uint64_t num_local_hist_bits;
  uint64_t num_pc_bits;
  uint64_t num_branch_hist_bits;
  uint64_t num_total_bits;
  uint8_t bits_per_weight;
  uint32_t theta_max;
  uint32_t size_per_perceptron;
  uint32_t num_perceptrons;
  int64_t** table;
  int64_t max_val;
  int64_t min_val;
} Perceptron;

typedef struct GSHARE_PRED {
  uint64_t ghr_bits;
  uint8_t *bht;
  uint32_t bht_counter_size;
  uint32_t bht_counter_half_mark;
} GSHARE_PRED;

PBP PBP_PRED;
Perceptron PERC;

void perc_init()
{
  PBP_PRED.ghr_redundant_1_num=4;
  PBP_PRED.ghr_redundant_1_to_use=20;

  PBP_PRED.ghr_redundant_2_num=0;
  PBP_PRED.ghr_redundant_2_to_use=0;

  PBP_PRED.ghr_redundant_3_num=0;
  PBP_PRED.ghr_redundant_3_to_use=0;

  PERC.size_budget = 84*1024;
  PERC.num_pc_bits = 0;
  PERC.num_branch_hist_bits = 62;
  PERC.num_local_hist_bits = 13;
  PERC.num_total_bits = PERC.num_branch_hist_bits + PERC.num_local_hist_bits + PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use + PBP_PRED.ghr_redundant_3_to_use + 1;
  PERC.bits_per_weight = 9;
  
  PERC.theta_max = 1.93*(PERC.num_total_bits-1) + 14;
  
  PERC.size_per_perceptron = PERC.num_total_bits*PERC.bits_per_weight;
  PERC.num_perceptrons = PERC.size_budget / PERC.size_per_perceptron;
  PERC.table = (int64_t**)malloc(PERC.num_perceptrons * sizeof(int64_t*));
  for(int i=0;i<PERC.num_perceptrons;i++)
    PERC.table[i] = (int64_t*)calloc(PERC.num_total_bits,sizeof(int64_t));
  PERC.max_val = (1<<(PERC.bits_per_weight-1))-1;
  PERC.min_val = -(1<<(PERC.bits_per_weight-1));


  PBP_PRED.choice_table_size = 10;
  PBP_PRED.chooser_counter_size = 2;
  PBP_PRED.chooser = (uint8_t *)malloc((1<<PBP_PRED.choice_table_size) * sizeof(uint8_t));
  PBP_PRED.chooser_counter_half_mark = (1<<(PBP_PRED.chooser_counter_size-1))-1;
  for (uint64_t i = 0; i < (1<<PBP_PRED.choice_table_size); i++)
    PBP_PRED.chooser[i] = PBP_PRED.chooser_counter_half_mark;

  PBP_PRED.ghr = 0;

  PBP_PRED.pht_table_size = 11;
  PBP_PRED.pht_bits = 13;
  PBP_PRED.pht = (uint32_t *)malloc((1<<PBP_PRED.pht_table_size) * sizeof(uint32_t));
  for (uint64_t i = 0; i < (1<<PBP_PRED.pht_table_size); i++)
    PBP_PRED.pht[i] = 0;

  PBP_PRED.lbht_counter_size = 2;
  PBP_PRED.lbht_counter_half_mark = (1<<(PBP_PRED.lbht_counter_size-1))-1;
  PBP_PRED.lbht = (uint8_t *)malloc((1<<PBP_PRED.pht_bits) * sizeof(uint8_t));
  for(uint64_t i=0;i<(1<<PBP_PRED.pht_bits);i++)
    PBP_PRED.lbht[i] = PBP_PRED.lbht_counter_half_mark;
}

uint8_t perc_predict(uint32_t pc)
{
  uint32_t perc_table_index = ((hashPC(pc,PBP_PRED.ghr,12))) % PERC.num_perceptrons;
  int64_t* perc_table_entry = PERC.table[perc_table_index];

  uint32_t pht_index = (pc>>2)&((1<<PBP_PRED.pht_table_size)-1);
  uint32_t pht_entry =  PBP_PRED.pht[pht_index];

  uint32_t pht_entry_lower_bits = pht_entry & ((1 << PBP_PRED.pht_bits) - 1);
  uint8_t lbht_pred = PBP_PRED.lbht[pht_entry_lower_bits];

  uint64_t ghr_filter = 1;
  uint64_t pht_filter = 1;
  uint64_t pc_filter = 1;
  uint64_t ghrr_filter1 = 1;
  uint64_t ghrr_filter2 = 1;
  uint64_t ghrr_filter3 = 1;
  int64_t prod=perc_table_entry[0]; // bias
  for(int i=1;i<PERC.num_total_bits;i++)
  {
    if(i<=PERC.num_branch_hist_bits)
    {
      if((PBP_PRED.ghr & ghr_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghr_filter <<= 1;
    }
    else if (i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits)
    {
      if((pht_entry & pht_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      pht_filter <<= 1;
    }
    else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits)
    {
      if((pc>>2 & pc_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      pc_filter <<= 1;
    }
    else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_1_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_1_to_use) -1 );
      if((tmp & ghrr_filter1)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter1 <<= 1;      
    }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_2_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_2_to_use) -1 );
      if((tmp & ghrr_filter2)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter2 <<= 1;      
    }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use + PBP_PRED.ghr_redundant_3_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_3_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_3_to_use) -1 );
      if((tmp & ghrr_filter3)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter3 <<= 1;      
    }
  }
  uint8_t perc_pred;
  if(prod>=0)
    perc_pred =  TAKEN;
  else
    perc_pred =  NOTTAKEN;


  uint32_t choice_table_index = hashPC(pc, PBP_PRED.ghr, PBP_PRED.choice_table_size);
  uint8_t choice = PBP_PRED.chooser[choice_table_index];
  uint8_t final_pred;

  if (choice<=PBP_PRED.chooser_counter_half_mark)
    final_pred = perc_pred;
  else
    final_pred = lbht_pred<=PBP_PRED.lbht_counter_half_mark?NOTTAKEN:TAKEN;

  return final_pred;
}


void train_perc(uint32_t pc, uint8_t outcome)
{
  uint32_t perc_table_index = (hashPC(pc,PBP_PRED.ghr,12)) % PERC.num_perceptrons;
  int64_t* perc_table_entry = PERC.table[perc_table_index];

  uint32_t pht_index = (pc>>2)&((1<<PBP_PRED.pht_table_size)-1);
  uint32_t pht_entry =  PBP_PRED.pht[pht_index];

  uint32_t pht_entry_lower_bits = pht_entry & ((1 << PBP_PRED.pht_bits) - 1);
  uint8_t lbht_pred = PBP_PRED.lbht[pht_entry_lower_bits]<=PBP_PRED.lbht_counter_half_mark?NOTTAKEN:TAKEN;

  uint64_t ghr_filter = 1;
  uint64_t pht_filter = 1;
  uint64_t pc_filter = 1;
  uint64_t ghrr_filter1 = 1;
  uint64_t ghrr_filter2 = 1;
  uint64_t ghrr_filter3 = 1;
  int64_t prod=perc_table_entry[0]; // bias
  for(int i=1;i<PERC.num_total_bits;i++)
  {
    if(i<=PERC.num_branch_hist_bits)
    {
      if((PBP_PRED.ghr & ghr_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghr_filter <<= 1;
    }
    else if (i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits)
    {
      if((pht_entry & pht_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      pht_filter <<= 1;
    }
    else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits)
    {
      if((pc>>2 & pc_filter)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      pc_filter <<= 1;
    }
    else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_1_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_1_to_use) -1 );
      if((tmp & ghrr_filter1)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter1 <<= 1;      
    }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_2_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_2_to_use) -1 );
      if((tmp & ghrr_filter2)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter2 <<= 1;      
    }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use + PBP_PRED.ghr_redundant_3_to_use)
    {
      uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_3_num);
      tmp = tmp & ((1<<PBP_PRED.ghr_redundant_3_to_use) -1 );
      if((tmp & ghrr_filter3)==0)
        prod -= perc_table_entry[i];
      else
        prod += perc_table_entry[i];
      ghrr_filter3 <<= 1;      
    }
  }

  uint8_t perc_pred;
  if(prod>=0)
    perc_pred =  TAKEN;
  else
    perc_pred =  NOTTAKEN;

  uint64_t percSteps = llabs(prod);


  uint32_t choice_table_index = hashPC(pc, PBP_PRED.ghr, PBP_PRED.choice_table_size);
  uint8_t choice = PBP_PRED.chooser[choice_table_index];

  // update choice predictor
  if(outcome == lbht_pred && outcome!=perc_pred && PBP_PRED.chooser[choice_table_index]!=(2*PBP_PRED.chooser_counter_half_mark+1))
      PBP_PRED.chooser[choice_table_index]++;
  else if (outcome == perc_pred && outcome!=lbht_pred && PBP_PRED.chooser[choice_table_index]!=0)
      PBP_PRED.chooser[choice_table_index]--;

  // update BHTs
  if (outcome == NOTTAKEN)
  {
    if(PBP_PRED.lbht[pht_entry_lower_bits]!=0)
      PBP_PRED.lbht[pht_entry_lower_bits]--;
  }

  if (outcome == TAKEN)
  {
    if(PBP_PRED.lbht[pht_entry_lower_bits]!=(2*PBP_PRED.lbht_counter_half_mark+1))
      PBP_PRED.lbht[pht_entry_lower_bits]++;
  }


  // update PERC
  int t;
  if(outcome==NOTTAKEN)
    t=-1;
  else
    t=1;

  if((perc_pred!=outcome) || (percSteps<=PERC.theta_max))
  {
    if(perc_table_entry[0]>PERC.min_val || perc_table_entry[0]<PERC.max_val)
       perc_table_entry[0] += t;

    uint64_t ghr_filter = 1;
    uint64_t pht_filter = 1;
    uint64_t pc_filter = 1;
    uint64_t ghrr_filter1 = 1;
    uint64_t ghrr_filter2 = 1;
    uint64_t ghrr_filter3 = 1;
    for(uint64_t i=1;i<PERC.num_total_bits;i++)
    {
      int to_add;
      if(i<=PERC.num_branch_hist_bits)
      {
        if(( ((PBP_PRED.ghr&ghr_filter)!=0) && (t==1)) || (((PBP_PRED.ghr&ghr_filter)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        ghr_filter <<= 1;
      }
      else if(i<=PERC.num_branch_hist_bits+PERC.num_local_hist_bits)
      {
        if(( ((pht_entry&pht_filter)!=0) && (t==1)) || (((pht_entry&pht_filter)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        pht_filter <<= 1;        
      }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits)
      {
        if(( (((pc>>2)&pc_filter)!=0) && (t==1)) || ((((pc>>2)&pc_filter)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        pc_filter <<= 1;     
      }
      else if (i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use)
      {
        uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_1_num);
        tmp = tmp & ((1<<PBP_PRED.ghr_redundant_1_to_use) -1 );
        if(( ((tmp&ghrr_filter1)!=0) && (t==1)) || (((tmp&ghrr_filter1)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        ghrr_filter1 <<= 1; 
      }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use)
      {
        uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_2_num);
        tmp = tmp & ((1<<PBP_PRED.ghr_redundant_2_to_use) -1 );
        if(( ((tmp&ghrr_filter2)!=0) && (t==1)) || (((tmp&ghrr_filter2)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        ghrr_filter2 <<= 1;         
      }
      else if(i<=PERC.num_branch_hist_bits + PERC.num_local_hist_bits+ PERC.num_pc_bits + PBP_PRED.ghr_redundant_1_to_use + PBP_PRED.ghr_redundant_2_to_use + PBP_PRED.ghr_redundant_3_to_use)
      {
        uint64_t tmp = PBP_PRED.ghr ^ (PBP_PRED.ghr>>PBP_PRED.ghr_redundant_3_num);
        tmp = tmp & ((1<<PBP_PRED.ghr_redundant_3_to_use) -1 );
        if(( ((tmp&ghrr_filter3)!=0) && (t==1)) || (((tmp&ghrr_filter3)==0) && (t==-1)))
          to_add = 1;
        else
          to_add = -1;
        ghrr_filter3 <<= 1;         
      }

      if(perc_table_entry[i]>PERC.min_val && perc_table_entry[i]<PERC.max_val)
        perc_table_entry[i] += to_add;

    }
  }

  PBP_PRED.ghr = ((PBP_PRED.ghr << 1) | outcome);
  PBP_PRED.pht[pht_index] = ((PBP_PRED.pht[pht_index] << 1) | outcome);
}


// **************************************
// **************************************


void init_predictor()
{
  switch (bpType)
  {
  case STATIC:
    break;
  case GSHARE:
    init_gshare();
    break;
  case CUSTOM:
    perc_init();
  default:
    break;
  }
}

uint32_t make_prediction(uint32_t pc, uint32_t target, uint32_t direct)
{

  // Make a prediction based on the bpType
  switch (bpType)
  {
  case STATIC:
    return TAKEN;
  case GSHARE:
    return gshare_predict(pc);
  case CUSTOM:
    return perc_predict(pc);
  default:
    break;
  }

  return NOTTAKEN;
}



void train_predictor(uint32_t pc, uint32_t target, uint32_t outcome, uint32_t condition, uint32_t call, uint32_t ret, uint32_t direct)
{
  if (condition)
  {
    switch (bpType)
    {
    case STATIC:
      return;
    case GSHARE:
      return train_gshare(pc, outcome);
    case CUSTOM:
      return train_perc(pc, outcome);
    default:
      break;
    }
  }
}
