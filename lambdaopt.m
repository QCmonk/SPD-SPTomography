function lam = lambdaopt(reconstruct, densitybasis)
% compute dimension of system
[dim, ~] = size(reconstruct{1});
%--------------------------------------------------------------------------
%-------------------------- STAGE 1 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% setup Lambda problem

A = zeros(dim^2);
b = zeros(dim^2);
for i=1:dim^2
    A(i,:) = reshape(densitybasis{i}.', [1,dim^2]);
    b(i,:) = reshape(reconstruct{i}, [1,dim^2]);
end

% Lamm is a two dimensional array with columns indexing input basis states 
% and rows indexing terms in the sum: sum_{k} lam_{jk}*rho_k
cvx_begin quiet
        cvx_precision best
        variable lam(dim^2, dim^2) complex
        minimise( norm(lam.'*A - b))
cvx_end

end