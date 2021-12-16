from tkinter import *
from tkinter import filedialog
from _thread import *
from math import *
import numpy as np
from PIL import ImageTk, Image
from sklearn.metrics import confusion_matrix
from skimage.filters import unsharp_mask, sato
from PIL import ImageFilter
import cv2

class picture():
    input = []
    inputMask = []
class Window(Frame):
    accuracy = float(0)
    sensitivity = float(0)
    specificity = float(0)
    mean_ar = float(0)
    mean_geo = float(0)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Wykrywanie naczyń dna siatkówki oka")
        self.pack(fill=BOTH, expand=True)
        self.grid(padx=4,pady=4)
        canvasSize=300

        self.inputCanvas = Canvas(self,width=canvasSize, height=canvasSize)
        self.inputCanvas.create_rectangle(2,2,canvasSize,canvasSize)
        self.inputCanvas.create_text(canvasSize/2,canvasSize/2,text="Obraz wejściowy")
        self.inputCanvas.grid(row=0,column=0)

        self.inputMaskCanvas = Canvas(self, width=canvasSize, height=canvasSize)
        self.inputMaskCanvas.create_rectangle(2, 2, canvasSize, canvasSize)
        self.inputMaskCanvas.create_text(canvasSize / 2, canvasSize / 2, text="Maska ekspercka wejściowa")
        self.inputMaskCanvas.grid(row=0, column=1)

        self.firstCanvas = Canvas(self, width=canvasSize,height=canvasSize)
        self.firstCanvas.create_rectangle(2,2,canvasSize,canvasSize)
        self.firstCanvas.create_text(canvasSize/2,canvasSize/2,text="Wygenerowana maska")
        self.firstCanvas.grid(row=0,column=2)

        self.outputCanvas = Canvas(self,width=canvasSize,height=canvasSize)
        self.outputCanvas.create_rectangle(2,2,canvasSize,canvasSize)
        self.outputCanvas.create_text(canvasSize/2,canvasSize/2,text="Naczynia zaznaczone na obrazie")
        self.outputCanvas.grid(row=0,column=3)

        self.uploadInputButton = Button(self,text="Wgraj obraz",command=self.upload_input_file, bg ="white")
        self.uploadInputButton.grid(row=1,column=0,pady=2)

        self.uploadInputButton = Button(self, text="Wgraj maske ekspercka", command=self.upload_input_mask, bg="white")
        self.uploadInputButton.grid(row=1, column=1, pady=2)

        xpadding=5
        Label(self, text="Treshold:").grid(row=3, column=0, sticky='w', padx=xpadding)
        self.thresholdEntry = Entry(self, width=4, justify=RIGHT)
        self.thresholdEntry.grid(row=3, column=0, sticky='e', padx=xpadding)

        self.closingVar = IntVar(value=1)
        Radiobutton(self,text="Nie wypełniaj naczyń",variable=self.closingVar,value=0).grid(row=3,column=1,sticky='n',padx=xpadding)
        Radiobutton(self,text="Wypełnij naczynia",variable=self.closingVar,value=1).grid(row=4,column=1,sticky='n',padx=xpadding)

        self.startButton = Button(self, text="Start", command=self.generatemask, width=20, height=2, bg='lightblue')
        self.startButton.grid(row=1, column=2, sticky='n', padx=20, pady=10)

        self.error = StringVar()
        Label(self, textvariable=self.error, fg="red", font=("Helvetica", 16)).grid(row=8)

        self.set_default_values()
        self.master.update()

    def set_default_values(self):
        self.thresholdEntry.insert(END, 25)

    def upload_input_file(self):
        filename = filedialog.askopenfilename()
        self.set_input_image(filename)
    def upload_input_mask(self):
        filename = filedialog.askopenfilename()
        self.set_input_mask(filename)

    def set_image(self,path,canvas):
        img = Image.open(path)
        print(img)
        img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.ANTIALIAS)
        picture.input = img
        canvas.image = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=canvas.image, anchor=NW)
    def set_mask(self,path,canvas):
        img = Image.open(path)
        print(img)
        img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.ANTIALIAS)
        picture.inputMask = img
        canvas.image = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=canvas.image, anchor=NW)

    def set_input_image(self,path):
        self.set_image(path,self.inputCanvas)
    def set_input_mask(self, path):
        self.set_mask(path,self.inputMaskCanvas)

    def generatemask(self):
        self.firstCanvas.create_rectangle(0, 0, self.firstCanvas.winfo_width(), self.firstCanvas.winfo_height(), fill="black")
        start_new_thread(self._generatemask, ())

    def _generatemask(self):
        pic = picture.input
        pic = pic.filter(ImageFilter.SHARPEN)
        pic = pic.convert('L')
        #pic = exposure.equalize_hist(pic)
        pic = pic.filter(ImageFilter.FIND_EDGES)
        pic = np.array(pic)

        th, pic = cv2.threshold(pic, int(self.thresholdEntry.get()), 255, cv2.THRESH_BINARY);

        pic=self.delete_boundary(pic,np.array(picture.input.convert('L')))

        pic=self.denoise(pic)

        if self.closingVar.get()==1:
            pic=self.morphologicClose(pic,3)

        self.setGenMask(Image.fromarray(pic))

        self.accuracy, self.sensitivity, self.specificity, self.mean_ar, self.mean_geo = self.analysis(np.array(pic), np.array(picture.inputMask))

        ogPicAr = np.array(picture.input)
        for i in range (0,ogPicAr.shape[0]):
            for j in range(0,ogPicAr.shape[1]):
                if pic[i][j]==255:
                    ogPicAr[i][j]=pic[i][j]
        self.setCombinedPic(Image.fromarray(ogPicAr))

        return pic

    def delete_boundary(self,pic,ogPic):
        th, bound_mask = cv2.threshold(ogPic, 0, 255, cv2.THRESH_BINARY);
        bound_mask = cv2.erode(bound_mask,np.ones((5, 5), np.uint8),iterations = 1)
        for i in range(0,pic.shape[0]):
            for j in range(0,pic.shape[1]):
                if(bound_mask[i][j]==0):
                    pic[i][j]=0
        return pic

    def denoise(self,pic):
        pic2=np.copy(pic)
        for i in range(1,pic.shape[0]-1):
            for j in range(1, pic.shape[1]-1):
                if pic[i][j]==255:
                    if pic[i-1][j-1]==0 and pic[i-1][j]==0 and pic[i-1][j+1]==0 and pic[i][j-1]==0 and pic[i][j+1]==0 and pic[i+1][j-1]==0 and pic[i+1][j]==0 and pic[i+1][j+1]==0:
                        pic2[i][j]=0
        return pic2

    def morphologicClose(self,pic,size):
        kernel = np.ones((size, size), np.uint8)
        pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel)
        thin = np.zeros(pic.shape, dtype='uint8')
        pic = cv2.bitwise_or(pic,thin)
        return pic

    def analysis(self, generated, input):
        cm = confusion_matrix(np.array(generated, dtype=bool).ravel(), np.array(input, dtype=bool).ravel())
        # print(cm)
        accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
        sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
        specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
        mean_ar = (sensitivity + specificity) / 2
        mean_geo = (sensitivity * specificity) ** (0.5)

        print("trafność = ", accuracy)
        print("czułość = ", sensitivity)
        print("swoistość = ", specificity)
        print("śr. arytmetyczna = ", mean_ar)
        print("śr. geometryczna = ", mean_geo)

        return accuracy, sensitivity, specificity, mean_ar, mean_geo


    def setGenMask(self,pic):
        self.firstCanvas.image = ImageTk.PhotoImage(pic)
        self.firstCanvas.create_image(0, 0, image=self.firstCanvas.image, anchor=NW)

    def setCombinedPic(self,pic):
        self.outputCanvas.image=ImageTk.PhotoImage(pic)
        self.outputCanvas.create_image(0, 0, image=self.outputCanvas.image, anchor=NW)

root = Tk()
root.geometry("1250x480")
app=Window(root)

root.mainloop()
