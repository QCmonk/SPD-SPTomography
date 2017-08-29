function densitym = statetomography(Eadiv, b) 
%#ok<*EQEFF>  
%#ok<*NOPRT>
beep off
% sort data structure issues of measurement vector
b = cell2mat(b)';

% get dimensions of system
len = length(b);
[~,dim] = size(Eadiv{1});
qubits = log2(dim);

% define spanning measurement set
Oi = zeros(dim, dim, dim^2-1);
for i=1:dim^2-1
   Oi(:,:,i) = Eadiv{i}; 
end

% construct operator space matrix, exploiting Pauli basis properties
A = dim*diag(ones(len,1),0);

% begin convex optimisation
cvx_begin 
    cvx_precision best
    variable x(len)
    minimise( norm(A*x - b) )
    subject to
        % positive definite requirement
        eye(dim)/(2^qubits) + reshape(Oi,[dim,len*dim]) * kron(x,eye(dim)) == hermitian_semidefinite(dim)
        % trace constraint
        trace(eye(dim)/(2^qubits) + reshape(Oi,[dim,len*dim]) * kron(x,eye(dim))) == 1
cvx_end
    
densitym=rhogen(Oi,x,dim);
end

% computes density matrix given coherence vector and spanning set fo N
% dimensional Hilbert space
function density = rhogen(measSet,r,dim) 

density = eye(dim)/(2^log2(dim));
rhoSeg = zeros(dim,dim,dim^2-1);

    for j=1:length(r)
        rhoSeg(:,:,j) = r(j)*measSet(:,:,j);
        density = density + rhoSeg(:,:,j);
    end
end