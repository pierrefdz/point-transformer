import matplotlib.pyplot as plt
import pickle

# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-3 without conv/accs_100epoch_transdim4.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-4 transdim 64/accs_100epoch_transdim64.p'
path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-3 transdim 64/accs_100epoch_transdim64.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-3 dataaugment transdim4/accs_100epoch_transdim4.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/SGD 0.05/accs_100epoch_transdim32.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/SGD 1e-3/accs_100epoch_transdim32.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-3 transdim 32/accs_100epoch_transdim32.p'
# path = r'C:/Users/pierr/Documents/Ponts-MVA/MVA/NPM3D/NPM3D_Projet/experiments/Adam 1e-3 without trans/accs_100epoch_transdim64.p'

train_accs, val_accs = pickle.load(open(path, 'rb'))

print(max(val_accs))
print()
print(len(train_accs))
print(len(val_accs))
print()

if len(train_accs)==100 and len(val_accs)==20:
    plt.figure()
    plt.plot(train_accs, label="Accuracy on train set")
    plt.plot([5*i for i in range(0,21)],[train_accs[0]-20]+val_accs, label="Accuracy on test set")
    plt.legend()
    plt.title('Max accuracy on the test set of: '+ str(max(val_accs))[0:5])
    plt.show()

if len(train_accs)==100 and len(val_accs)==10:
    plt.figure()
    plt.plot(train_accs, label="Accuracy on train set")
    plt.plot([10*i for i in range(0,11)],[train_accs[0]-10]+val_accs, label="Accuracy on test set")
    plt.legend()
    plt.title('Max accuracy on the test set of: '+ str(max(val_accs))[0:5])
    plt.show()
