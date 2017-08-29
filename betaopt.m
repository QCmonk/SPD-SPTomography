function beta = betaopt(densitybasis, operatorbasis, partition)
% compute dimension of system
[dim, ~] = size(densitybasis{1});
%--------------------------------------------------------------------------
%-------------------------- STAGE 2 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% extract partition range
par1 = partition(1);
par2 = partition(2);
    
% flatten operator
rho = zeros(dim^2);
for i=1:dim^2
   rho(i,:) = reshape(densitybasis{i}.', [1,dim^2]);
end

% construct constrained basis set acting on basis states baserho(m,j,n, rho)
baserhobase = zeros(dim^4);

for m=1:dim^2
    for j=1:dim^2
        for n=1:dim^2
            [r,s] = subget(m,j,n,1:dim^2,dim);
            baserhobase(s,r) = reshape((operatorbasis{m}*densitybasis{j}*operatorbasis{n}').', [1,dim^2]);
        end
    end
end

% preallocate return matrix
beta = zeros(dim^4);

% compute section of beta matrix
for j=par1:par2
    ind = (j-1)*dim^2 +[1:dim^2];
    cvx_begin quiet
        cvx_precision high
        variable betab(dim^4, dim^2) complex
        minimise( norm(betab*rho - baserhobase(:,ind)))
    cvx_end
    beta(ind,:) = betab.';
end
end