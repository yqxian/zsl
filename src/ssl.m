function label = ssl(W, Y, alpha)

d = 1 ./ sqrt(sum(W,2));
D = diag(d);

S = D * W * D;

max_ite = 1000;
eps = 1e-10;

F_old = Y;
F = [];
for i=1:max_ite
    F = alpha * S * F_old + (1 - alpha) * Y;
    res = sum(sum((F - F_old).^2));
    disp(['Iteration ' num2str(i) ': error is ' num2str(res)]);
    if( res < eps)
        break;
    end
    F_old = F;
end

[score, label] = max(F');