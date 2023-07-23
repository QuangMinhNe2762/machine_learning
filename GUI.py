from tkinter import *
from tkinter import messagebox as msb
import matplotlib.pyplot as plt
import function as fc
import time

window = Tk()
window.title("demo")
window.geometry("1500x300")


def nhap_du_lieu():
    data = fc.lay_du_lieu_hl()
    data.head()
    if len(data) > 0:
        msb.showinfo("Message", f"lấy dữ liệu huấn luyện thành công\ncó {len(data)} dữ liệu")
    else:
        msb.showinfo("Message", "lấy dữ liệu thất bại")


def hien_thi_hinh_anh_random():
    ten_hinh_anh = fc.lay_ten_hinh_anh()
    photo = plt.imread(fc.path_hinh_anh)
    plt.imshow(photo)
    plt.show()


def huan_luyen_mo_hinh():
    sl = fc.lay_toa_do_doi_tuong()
    if sl > 0:
        time.sleep(6)
        msb.showinfo("Message", "huấn luyện thành công")
    else:
        msb.showinfo("Message", "huấn luyện thất bại không có dữ liệu để huấn luyện")


def huan_luyen_random():
    if fc.lay_toa_do_doi_tuong() > 0:
        rdnumber = fc.random_doi_tuong_hl()
        hinh_anh = fc.hien_thi_doi_tuong_random(rdnumber)
        plt.imshow(hinh_anh)
        plt.title("ảnh gốc")
        plt.figure()
        hinh_anh_phat_hien_doi_tuong = fc.phat_hien_doi_tuong_bang_du_lieu_HL(rdnumber)
        plt.imshow(hinh_anh_phat_hien_doi_tuong)
        plt.title("ảnh huấn luyện")
        plt.show()


def nhap_du_lieu_test():
    sl = fc.sl_du_lieu_test()
    if sl > 0:
        msb.showinfo("Message", f"nhập dữ liệu test thành công có {sl} dữ liệu")
    else:
        msb.showinfo("Message", "nhập dữ liệu test thất bại")


def cach_rcnn_hd():
    rdnumber = fc.random_doi_tuong_hl()
    anh_goc = fc.anh_rcnn_hoatDong(rdnumber)
    plt.imshow(anh_goc)
    plt.figure()
    anh_phdt, sl_doiTuong = fc.rcnn_hoatDong(rdnumber)
    plt.imshow(anh_phdt)
    plt.title(f"số lượng đối tượng phát hiện:{sl_doiTuong}")
    plt.show()


def thuc_hien_test():
    img = fc.lay_hinh_anh_test()
    plt.imshow(img)
    plt.title("ảnh gốc")
    img1, tylercnn, tyleyolo, str, str1, slxe,slcoxe = fc.test_rcnn(img)
    if len(tylercnn) > 0:
        resultryolo = sum(tyleyolo) / len(tyleyolo)
        resultrcnn = sum(tylercnn) / slcoxe
        danhgia = Tk()
        danhgia.title("đánh giá")
        danhgia.geometry("1500x600")
        labeltitle = Label(
            danhgia,
            width=40,
            font=("Time New Roman", 20),
            fg="black",
            text="đánh giá tỷ lệ của 2 thuật toán yolo và RCNN",
            bg="aqua",
        )
        labeltitle.config(height=5, width=50)
        labeltitle.grid(column=3, row=0)

        strsl = "số lượng xe phát hiện là ", slxe
        labelslxe = Label(
            danhgia, width=40, font=("Time New Roman", 10), fg="black", text=strsl
        )
        labelslxe.config(height=5, width=50)
        labelslxe.grid(column=1, row=3)

        labelsldt = Label(
            danhgia, width=40, font=("Time New Roman", 10), fg="black", text=str
        )
        labelsldt.config(height=5, width=50)
        labelsldt.grid(column=1, row=1)

        labelpl = Label(
            danhgia, width=40, font=("Time New Roman", 10), fg="black", text=str1
        )
        labelpl.config(height=5, width=50)
        labelpl.grid(column=1, row=2)

        stringresultrcnn = "tỷ lệ dự đoán đối tượng của RCNN:", resultrcnn*100
        labelrcnn = Label(danhgia,width=40,font=("Time New Roman", 10),fg="black",text=stringresultrcnn)
        labelrcnn.config(height=5, width=50)
        labelrcnn.grid(column=1, row=4)

        stringresultyolo = "tỷ lệ dự đoán đối tượng của YOLO:", resultryolo * 100
        labelpl = Label(
            danhgia,
            width=40,
            font=("Time New Roman", 10),
            fg="black",
            text=stringresultyolo,
        )
        labelpl.config(height=5, width=50)
        labelpl.grid(column=1, row=5)

        plt.figure()
        plt.imshow(img1)
        plt.title("ảnh kiểm tra")
        plt.text(20,1, "RCNN", fontsize=12, color='red', ha='center')
        plt.text(100,1, "YOLO", fontsize=12, color='green', ha='center')
        plt.show()
        danhgia.mainloop()
    else:
        msb.showinfo("Message","ảnh không có xe")


label = Label(window,width=40,font=("Time New Roman", 20),fg="black",text="PHÁT HIỆN ĐỐI TƯỢNG XE HƠI",bg="aqua")
label.config(height=5, width=25)
label.grid(column=3, row=0)

btnnhapDL = Button(window, text="nhập dữ liệu huấn luyện", command=nhap_du_lieu)
btnnhapDL.config(height=5, width=25)
btnnhapDL.grid(column=1, row=1)

btn_hL = Button(window, text="huấn luyện mô bằng dữ liệu", command=huan_luyen_mo_hinh)
btn_hL.config(height=5, width=25)
btn_hL.grid(column=2, row=1)

btnhuanluyen = Button(window, text="random hình ảnh đã huấn luyện", command=huan_luyen_random)
btnhuanluyen.config(height=5, width=25)
btnhuanluyen.grid(column=3, row=1)

btn_test = Button(window, text="nhập dữ liệu kiểm tra", command=nhap_du_lieu_test)
btn_test.config(height=5, width=25)
btn_test.grid(column=5, row=1)

btn_rcnnhl = Button(window, text="cách RCNN hoạt động", command=cach_rcnn_hd)
btn_rcnnhl.config(height=5, width=25)
btn_rcnnhl.grid(column=4, row=1)

btn_kiemtra = Button(
    window, text="random hình ảnh đã phát hiện đối tượng", command=thuc_hien_test
)
btn_kiemtra.config(height=5, width=50)
btn_kiemtra.grid(column=6, row=1)

window.mainloop()
