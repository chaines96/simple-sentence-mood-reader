import torch, numpy, pandas, sklearn
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from tkinter import * #for the GUI

#categories_2 = {"Happy":numpy.float32(0), "Sad":numpy.float32(1), "Angry":numpy.float32(2), "Informative":numpy.float32(3), "Nonsense":numpy.float32(4), "Funny":numpy.float32(5)} 
categories_2 = {"Happy":0, "Sad":1, "Angry":2, "Informative":3, "Nonsense":4, "Funny":5} #The hard coded results of output. This dict will convert to a list in testing. #The hard coded results of output. This dict will convert to a list in testing.

#Setting some variables
EPOCHS = 100 #Amount of times to iterate over the training data.
word_len = 20

class SentenceDataset(Dataset):
    def __init__(self, file_path, label_path, transform=None,target_transform=None):
        pandas.options.display.max_rows = 100 #Panda's default is 60
        try:
            #self.data = pandas.read_csv(file_path)
            data_textlines = open(file_path, 'r', encoding='utf-8').read().splitlines()
            label_textlines = open(label_path, 'r', encoding='utf-8').read().splitlines()
            data_floats = list()
            #label_loats = list()
            for item in data_textlines:
                float_array = [numpy.float32(0)]*word_len
                word_index = 0 
                for word in item.split(' '):
                    hashed_word = numpy.float32(0) 
                    indexe = 0
                    expo = 1
                    for letter in word: #'toy' hashing function
                        hashed_word = hashed_word + numpy.float32(ord(letter)*expo)
                        expo = expo*10
                        indexe = indexe + 1
                    if (word_index < word_len):
                        float_array[word_index] = hashed_word #word added to sentense
                    word_index = word_index + 1
                data_floats.append(float_array) #sentense is converted to list of floats.
            labels_floats = list()
            for item in label_textlines:
                labels_floats.append(categories_2[item])
            #for item in label_textlines:
            #    label_floats.append(float(item))
            self.data = torch.as_tensor(data_floats)
            self.labels = torch.as_tensor(labels_floats)
        except Exception as e:
            #print("Cannot read or write to %s",file_path)
            print(e)
            exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return torch.from_numpy(self.data.iloc[idx].astype(numpy.float32).to_numpy()), self.labels.iloc[idx, 0]
        return self.data[idx], self.labels[idx] #Returns an array with one data entry and an integer representing its label.

training_data = SentenceDataset(
    file_path="data.txt",
    label_path = "data_labels.txt"
)

test_data = SentenceDataset(
    file_path="test.txt",
    label_path = "test_labels.txt"
)

#Two variables that must be defined after we verify that data.csv and test.csv are loaded.
data_len = len(open("data.txt", 'r', encoding='utf-8').read().splitlines()) #The amount of rows in the CSV file, representing the number of entries of the training data.
batch_size = (data_len // 4) + 1 #The size of the batch of each epoch. Should sacle with the size of the data; I would consider hard coding this to 5.

# Data loaders prototyped from PyTorch.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Initializing the NeutralNetwork class.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim = 0, end_dim = 0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(word_len, 64), #Sends each column of the batch number's row to 64 input neurons in a linear layer.
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6) #The output neurons here represent labels.
        )

    def forward(self, x):
        #x = self.flatten(x) 
        x = torch.flatten(x, start_dim = 0, end_dim = 0)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# This try block will load load a model to write to it
try:
    model.load_state_dict(torch.load("neural_net.pth", weights_only=True))
except:
    print("neural_net.pth not found, creating a new neural network.")

#Loss Function
loss_func = nn.CrossEntropyLoss()
optimize_func = torch.optim.SGD(model.parameters(), lr=1e-2)

#Training Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #The error in the prediction
        predicted = model(X)
        loss = loss_func(predicted, y)

        #Backpropagation
        loss.backward()
        optimize_func.step()
        optimize_func.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return

#Testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predicted = model(X)
            test_loss += loss_func(predicted, y).item()
            correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"____TEST ERROR______: \n Total Accuracy: {(100*correct):>0.1f}%, Averge Loss: {test_loss:>8f} \n")
    return

#Calls the training function for set number of epochs
for i in range(EPOCHS):
    print(f"Epoch #{i+1}\n________")
    train(train_dataloader, model, loss_func, optimize_func)
    test(test_dataloader, model, loss_func)
print("Completd training. Saving to neural_net.pth.")

torch.save(model.state_dict(), "neural_net.pth")
print("Saved PyTorch Model State to neural_net.pth.")

categories = ["Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny"]

#This next block uses the tkinter library to create a GUI.
main_window = Tk(screenName=None, baseName=None, className=' Result', useTk=1)
main_window.geometry("500x200") #Sets the result output window's size.
main_window.resizable(True, True)

model.eval() #Turns our neural network into evaluation mode.
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x) #Returns a tensor with predictions. Should have as many values as there are classes.
    result = categories[torch.argmax(pred, dim=0)] #Returns the class corresponding ot the index of the maximum value of the tensor (i.e. what the neural net thinks is most likely).
    out_text = f'Predicted: "{result}", Actual: "{categories[int(y)]}"'

model.train()  #Turns the neural network into training mode.

new_sentence = "" #Defining the new_sentence variable as a string.

def enter_data():
    new_sentence = new_sentence_var.get() #The string will be euqal to the text the user has entered.
    return

enter_button = Button(main_window,text="Progress",command=enter_data,activebackground="blue", activeforeground="white",anchor="center",bd=3,bg="lightgray",cursor="hand2",
                      disabledforeground="gray",fg="black",font=("Arial", 12),height=2,highlightbackground="black",highlightcolor="green",highlightthickness=2,justify="center",
                      overrelief="raised",padx=10,pady=5,width=15,wraplength=100)
new_sentence_var = StringVar()
new_sentence_var.set("")
text_entry = Entry(main_window, textvariable=new_sentence_var,font=('Arial',16))

#Creates a widget in the main_window tkinter instance.
result_widget = Label(main_window, text = out_text,font="Arial 16")
enter_button.pack(padx=20, pady=20)
text_entry.pack(padx=40, pady=20)
result_widget.pack()
main_window.mainloop()