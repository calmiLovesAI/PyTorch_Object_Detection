import cv2
import os
import shutil


class Video:
    def __init__(self, cfg, model, device, video_dir, write_video_dir, tmp_dir="./temp.jpg", pipeline_func=None):
        """

        :param cfg: 配置文件
        :param model: 模型
        :param device: 运行设备
        :param video_dir: 需要检测的视频路径
        :param write_video_dir: 生成的新视频路径
        :param tmp_dir: 临时帧保存位置
        :param pipeline_func: 检测单张图片的函数
        """
        self.cfg = cfg
        self.model = model
        self.device = device
        self.video_dir = video_dir
        self.write_video_dir = write_video_dir
        self.tmp_dir = tmp_dir
        self.pipeline_func = pipeline_func

    def show(self):
        capture = cv2.VideoCapture(self.video_dir)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_cnt = 0  # 计数第几帧
        while True:
            ret, frame = capture.read()
            if ret:
                frame_cnt += 1
                print("Processing frame {}".format(frame_cnt))
                cv2.imwrite(self.tmp_dir, frame)
                new_frame = self.pipeline_func(self.cfg, self.model, self.tmp_dir, self.device, print_on=False,
                                               save_result=False)
                cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("detect result", new_frame)
                cv2.waitKey(int(1000 / fps))
                os.remove(self.tmp_dir)
            else:
                break

        capture.release()
        cv2.destroyAllWindows()

    def write(self):
        frame_save_path = "./frames/"
        # 创建临时文件夹
        if not os.path.exists(frame_save_path):
            os.mkdir(frame_save_path)

        video_capture_1 = cv2.VideoCapture(self.video_dir)
        frame_counter = 0
        while True:
            ret, frame = video_capture_1.read()
            if ret:
                frame_counter += 1
                # 存储每一帧的图片
                cv2.imwrite(frame_save_path + "frame_{}.jpg".format(frame_counter), frame)
                if frame_counter % 50 == 0:
                    print("====================>Storing frame: %d<==================" % (frame_counter))
            else:
                break

        # 读取视频
        video_capture_2 = cv2.VideoCapture(self.video_dir)
        # 码率
        fps = video_capture_2.get(cv2.CAP_PROP_FPS)
        # 帧尺寸
        frame_size = (
            int(video_capture_2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture_2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(self.write_video_dir, fourcc, fps, frame_size)
        num_of_frame = 0  # 帧数
        while True:
            sucess, frame = video_capture_2.read()
            if sucess:
                num_of_frame += 1
                # 对每一帧图片进行处理
                frame_dir = frame_save_path + "frame_{}.jpg".format(num_of_frame)
                new_frame = self.pipeline_func(self.cfg, self.model, frame_dir, self.device, print_on=False,
                                               save_result=False)

                # 写入帧
                print("===================>Writing frame: %d<==================" % (num_of_frame))
                video_writer.write(new_frame)
            else:
                print("视频已生成！")
                break

        shutil.rmtree(frame_save_path)
