data {
  real<lower=1> segments; //number of markings (1 implies no marks, 2 implies 3 marks at - 0, 50%, 100% ) r[n] n = segments
  int<lower=1> n; //number of obervations
  //int<lower=1> total; //100% oflf the proportion, also = Rn
  vector[n] correct_ans; 
  vector[n] response; //participant's response
}

parameters {
  real<lower=0> b;
  real<lower=0> phi;
}

transformed parameters {
  vector[n] p; //estimate of person's response
  vector[n] riminus1_value;
  vector[n] ri_value;
  
  
  for (i in 1:n){
    //figure out which segment correct_ans lies in 
    riminus1_value[i] = floor(correct_ans[i] * segments) * (1/segments);
    ri_value[i] = riminus1_value[i] + 1/segments;
      
    p[i] = (pow((correct_ans[i] - riminus1_value[i]),b)/(pow((correct_ans[i] - riminus1_value[i]),b)+pow((ri_value[i] - correct_ans[i]),b))) * (1/segments) + (riminus1_value[i]);
  }
  
} 

model {
  //prior for phi,b
  phi ~ cauchy(0,5);
  b ~ lognormal(0,1);
  
  //model
  for (i in 1:n){
    response[i] ~ beta(p[i]*phi, (1-p[i])*phi);
  }
}

