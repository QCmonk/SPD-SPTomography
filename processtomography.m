function chi = processtomography(reconstruct, densitybasis, operatorbasis)
beep off
% compute dimension of system
[dim, ~] = size(reconstruct{1});
%--------------------------------------------------------------------------
%-------------------------- STAGE 1 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% setup Lambda problem

A = zeros(dim^2);
b = zeros(dim^2);
for i=1:dim^2
    A(i,:) = reshape(densitybasis{i}, [dim^2,1]);
    b(i,:) = reshape(reconstruct{i}, [dim^2,1]);
end

% Lamm is a two dimensional array with columns indexing input basis states 
% and rows indexing terms in the sum: sum_{k} lam_{jk}*rho_k
lamm = zeros(dim^2, dim^2);
for j=1:dim^2
    b = reshape(reconstruct{j}.', [1,dim^2]);
    cvx_begin quiet
        cvx_precision default
        variable lam(dim^2) complex
        minimise( norm(lam.'*A - b))
    cvx_end
    lamm(1:dim^2,j) = lam;
end
lamm
%--------------------------------------------------------------------------
%-------------------------- STAGE 2 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% Compute Beta matrix
    
% flatten operators
op = zeros(dim^2);
rho = zeros(dim^2);
for i=1:dim^2
   op(:,i) = reshape(operatorbasis{i}.', [dim^2,1]);
   rho(i,:) = reshape(densitybasis{i}.', [1,dim^2]);
end

% construct constrained basis set acting on basis states baserho(m,j,n, rho)
baserhobase = zeros(dim^4);

for m=1:dim^2
    for j=1:dim^2
        for n=1:dim^2
            [r,s] = subget(m,j,n,1:dim^2, dim);
            baserhobase(r,s) = reshape((operatorbasis{m}*densitybasis{j}*operatorbasis{n}).', [dim^2,1]);
        end
    end
end

% construct the gamma matrix
gam = blkdiag(rho,rho,rho,rho);
% optimise in one hit (mission and a half)
cvx_begin quiet
    cvx_precision default
    variable betab(dim^4,dim^4) complex
    minimise( norm(betab*gam - baserhobase.'))
cvx_end   
%--------------------------------------------------------------------------
%-------------------------- STAGE 3 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% compute the Chi matrix

% compute base products for use in constraints
basecont = [];
for m=1:dim^2
    for n=1:dim^2
        % pre allocate this in future version, will need to sort out indexing
        basecont = vertcat(basecont, operatorbasis{n}*operatorbasis{m});
    end
end

cvx_begin
    cvx_precision default
    variable chi(dim^2,dim^2) hermitian semidefinite
    minimise( norm( (pinv(full(betab).')*reshape(lamm, [dim^4,1])) - reshape(chi, [dim^4,1])))
    subject to
        % trace preserving (or not) constraint (pain in the neck to do in one step...CVXXXXX! *shakes fist*)
        real(repmat([1,0;0,1],1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]),dim).*basecont)) <= eye(dim);
        imag(repmat([1,0;0,1],1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]),dim).*basecont)) == 0;
        % trace relation between chi and P
        trace(reshape(reshape(chi, [dim^4,1]), [dim^2, dim^2])) == trace(repmat([1,0;0,1],1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]), dim).*basecont))/2;
        % matrix is positive semi definite
cvx_end



% reshape Chi matrix
chi = reshape(chi, [dim^2,dim^2]);
cptp(chi, operatorbasis, [1 0; 0 0])
end