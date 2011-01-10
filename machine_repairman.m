%lambda = 20 %arrival rate

miu = 100 %service rate
m = 8 %number of tokens
file1 = fopen('output','w')
%fix miu, vary lambda
for lambda = 10:10:100,%floor(miu) - 1,
    
    rho = lambda/miu
    x = zeros(1,m+1)
    for k = 1:m+1, x(k) = (rho^(k-1))*(factorial(m)/factorial(m-k+1)) ; end
    pi0 = 1/sum(x)
    response = (m/(miu*(1-pi0)) - 1/lambda)
    %fprintf('lambda = %5.5f',lambda)
    fprintf('response_time = %10.5f\n',response)
    fprintf(file1,'%10.5f\n',response)
end
fclose(file1)
