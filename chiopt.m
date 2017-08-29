function chi = chiopt(beta, operatorbasis, lam)
% compute dimension of system
[dim, ~] = size(operatorbasis{1});
% perform required data conversion


%--------------------------------------------------------------------------
%-------------------------- STAGE 3 OPTIMISATION---------------------------
%--------------------------------------------------------------------------
% compute the Chi matrix

% compute base products for use in constraints
basecont = [];
for m=1:dim^2
    for n=1:dim^2
        % pre allocate this in future version, will need to sort out indexing
        basecont = vertcat(basecont, operatorbasis{n}'*operatorbasis{m});
    end
end

cvx_begin
    cvx_precision best
    variable chi(dim^2,dim^2) hermitian semidefinite
    minimise( norm( (pinv(beta)*reshape(lam, [dim^4,1])) - reshape(chi, [dim^4,1])))
    subject to
        % trace preserving (or not) constraint (pain in the neck to do in one step...CVXXXXX! *shakes fist*)
        real(repmat(eye(dim),1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]),dim).*basecont)) <= eye(dim);
        imag(repmat(eye(dim),1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]),dim).*basecont)) == 0;
        % trace relation between chi and P
        trace(chi) == trace(repmat(eye(dim),1,dim^4)*(interleave(repmat(reshape(chi, [dim^4,1]),[1,dim]),dim).*basecont))/dim;
cvx_end
% reshape Chi matrix
chi = full(chi);
end