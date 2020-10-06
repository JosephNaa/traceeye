package tech.goqba.traceeye.pulse;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.PersistableBundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.traceeye.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class PulseActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2 {

    //카메라 사용
    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar CHEEK_RECT_COLOR = new Scalar(255, 0, 0, 255);
    private CascadeClassifier mJavaDetector;
    private CameraBridgeViewBase mOpenCvCameraView;
    private float mRelativeFaceSize = 0.3f;
    private int mAbsoluteFaceSize = 0;
    ppgWorkerThread thread;

    //입력 색상 데이터
    private Mat mRgba;
    private Mat mRgbaT;
    private Mat mRgbaF;
    private Mat mGray;
    private Mat cheek;

    //색상 값 변수
    private double r = 0;
    private double g = 0;
    private double b = 0;
    private double cg = 0;

    private double Cg_avg = 0;
    private double Cg_sum = 0;

    //얼굴(오볼) 사이즈
    private int fx;
    private int fy;
    private int fw;
    private int fh;

    //얼굴(왼볼) 사이즈
    private int cx;
    private int cy;
    private int cw;
    private int ch;

    //색상 값 배열 처리
    double[] Cg_Left_arr;
    List<Double> Cg_Left = new ArrayList<>();
    List<Double> Cg_nomal = new ArrayList<>();

    private final Queue<Double> tlx_queue = new LinkedList<Double>();
    private final Queue<Double> tly_queue = new LinkedList<Double>();
    private final Queue<Double> brx_queue = new LinkedList<Double>();
    private final Queue<Double> bry_queue = new LinkedList<Double>();

    //얼굴 인식 넓이-높이
    private Mat forehead;
    private Mat mTemp;
    private int widthR;
    private int heightR;

    //측정시간 표시
    static boolean running = true;
    static boolean onWorking = true;

    //프레임 처리
    private static int fps = 0;
    private int count = 0;

    //토스트 기능
    private Toast toast;
    public static int pulseRate = 0;

    //텍스트 출력 기능
    public static TextView output_pulse;

    //음성 파일 재생
    private static MediaPlayer mp, mp1, mp2, mp3, mp4;

    //측정 시간 카운트
    int time = 12;
    Handler handler;
    Handler mHandler = new Handler(Looper.getMainLooper());

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        //      lbpcascade_frontalface.xml           haarcascade_frontalface_alt2.xml
                        File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        mJavaDetector.load(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier for face");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pulse);
        checkPermission();

        int permissonCheck= ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA);

        if(permissonCheck == PackageManager.PERMISSION_GRANTED) {
            //카메라 설정
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view_be);
            //카메라 해상도 설정부
            mOpenCvCameraView.setMaxFrameSize(320, 240);
            mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
            mOpenCvCameraView.setCvCameraViewListener(this);
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

            //TextView 설정
            output_pulse = (TextView) findViewById(R.id.output_pulse);

            //음성 삽입부
            mp = MediaPlayer.create(PulseActivity.this, R.raw.start);
            mp1 = MediaPlayer.create(PulseActivity.this, R.raw.start2);
            mp2 = MediaPlayer.create(PulseActivity.this, R.raw.result);
            mp3 = MediaPlayer.create(PulseActivity.this, R.raw.restart);
            mp4 = MediaPlayer.create(PulseActivity.this, R.raw.countdown);

            if (!OpenCVLoader.initDebug()) {
                Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
            } else {
                Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }

            Thread myThread = new Thread(new Runnable() {
                public void run() {
                    while (true) {
                        try {
                            Thread.sleep(1000);
                            mOpenCvCameraView.enableView();
                        } catch (Throwable t) {
                        }
                    }
                }
            });

            //토스트 시작
            toast = Toast.makeText(PulseActivity.this, "화면을 바라보세요.", Toast.LENGTH_LONG);
            ViewGroup group = (ViewGroup) toast.getView();
            TextView messageTextView = (TextView) group.getChildAt(0);
            messageTextView.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 15);
            toast.show();
            mp.start();

            //측정 시작
            myThread.start();
        } else {
            recreate();
        }
    }
    //권한 체크
    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.READ_PHONE_STATE,
                            Manifest.permission.CAMERA,
                            Manifest.permission_group.CAMERA,
                            Manifest.permission_group.STORAGE,
                    }
                    , 1);
        }

    }

    //카메라 구동
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(height, height, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    //영상에서 Frame 검출하는 코드
    @Override
    public Mat onCameraFrame(JavaCameraView.CvCameraViewFrame inputFrame) {
        //영상에서 RGB 입력 코드
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        mTemp = inputFrame.rgba();
        widthR = mRgba.width();
        heightR = mRgba.height();

        //얼굴 인식 코드, time 변수가 0이 되면 맥박값 산출
        if (time == 0) {
            Cg_Left_arr = new double[Cg_Left.size()];
            for (int i = 0; i < Cg_Left.size(); i++) {
                Cg_Left_arr[i] = Cg_Left.get(i);
            }

            //Cg 색상 데이터 평균값 산출
            Cg_avg = Cg_sum / Cg_Left.size();
            fps = Cg_Left.size() / 10;

            //촬영 완료 후, 정규화 및 기타 측정치
            for (int i = 0; i < Cg_Left.size() - 1; i++) {
                double c = Cg_Left.get(i) - Cg_avg;
                Cg_nomal.add(c);
            }

            //생체 신호 측정부
            getBioSignal();

            //배열 초기화
            Cg_Left.clear();
            Cg_nomal.clear();
            time = time -1;

            //time 변수가 11이 되면 측정 시작
        } else if(time == 11){
            mHandler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(PulseActivity.this, "측정을 시작합니다.", Toast.LENGTH_LONG).show();
                    mp1.start();
                    handler = new Handler() {
                        public void handleMessage(Message msg) {
                            if (msg.what == 1) {
                                output_pulse.setTextColor(Color.BLACK);
                                output_pulse.setText("측정중 : " + msg.arg1);
                                mp4.start();
                            } else if (msg.what == 2){
                                mHandler.postDelayed(new Runnable() {
                                    @Override
                                    public void run() {
                                        Toast.makeText(PulseActivity.this, "측정이 완료되었습니다. \n 결과를 확인하세요.", Toast.LENGTH_LONG).show();
                                        mp2.start();
                                    }
                                }, 0);
                                time = 18;
                                count = 0;

                                output_pulse.setText("맥박 : " + Math.round(pulseRate) + "회");
                                output_pulse.setTextColor(Color.RED);

                                Cg_Left.clear();
                                Cg_nomal.clear();
                            }
                        }
                    };

                    //카운트 시작
                    thread = null;
                    thread = new ppgWorkerThread(handler);
                    running = true;
                    thread.start();
                    onWorking = true;
                }
            }, 0);

            time = time - 1;

            //time 변수가 10보다 작으면 색상 데이터 평균값 계산
        } else if (time <= 10) {

            MatOfRect faces = new MatOfRect();
            if (mAbsoluteFaceSize == 0) { //사이즈 설정
                long height = mGray.rows();
                if (Math.round(height * mRelativeFaceSize) > 0)
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }

            Mat mGray = new Mat(mRgba.width(), mRgba.height(), CvType.CV_8UC1);
            Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_BGR2GRAY);

            Core.flip(mGray.t(), mGray, 0);

            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

            fx = 0;
            fy = 0;
            fw = 0;
            fh = 0;

            //얼굴 검출 코드
            Rect[] facesArray = faces.toArray();
            for (Rect aFacesArray : facesArray) {

                //평균필터를 이용한 얼굴 검출 안정화
                if (tlx_queue.size() < 15) {
                    tlx_queue.offer(mRgba.width() - aFacesArray.tl().y);
                    tly_queue.offer(aFacesArray.tl().x);
                    brx_queue.offer(mRgba.width() - aFacesArray.br().y);
                    bry_queue.offer(aFacesArray.br().x);
                } else {
                    tlx_queue.offer(mRgba.width() - aFacesArray.tl().y);
                    tly_queue.offer(aFacesArray.tl().x);
                    brx_queue.offer(mRgba.width() - aFacesArray.br().y);
                    bry_queue.offer(aFacesArray.br().x);

                    double tlx = get_avg(tlx_queue.toArray());
                    double tly = get_avg(tly_queue.toArray());
                    double brx = get_avg(brx_queue.toArray());
                    double bry = get_avg(bry_queue.toArray());

                    Imgproc.rectangle(mRgba, new Point(tlx, tly), new Point(brx, bry), FACE_RECT_COLOR, 2);

                    //볼(하) 검출 코드;
                    cx = (int) (((tlx - brx) * 0.35) + brx) - 10;
                    cy = (int) (((bry - tly) * 0.72) + tly);
                    cw = 10;
                    ch = 10;

                    // 볼 확인 코드
                    //Imgproc.rectangle(mRgba, new Point(cx, cy), new Point(cx+cw, cy+ch), CHEEK_RECT_COLOR, 2);
                    Rect roi = new Rect(cx, cy, cw, ch); // 좌측 볼 부분 만큼 이미지 자르기
                    cheek = new Mat(mRgba, roi);

                    //볼(상) 검출 싸이즈 설정
                    fx = (int) (((tlx - brx) * 0.35) + brx) + 5;
                    fy = (int) (((bry - tly) * 0.72) + tly) - 2;
                    cg = 0;
                    //Imgproc.rectangle(mRgba, new Point(fx, fy), new Point(fx+cw, fy+ch), CHEEK_RECT_COLOR, 2);

                    //관심 영역 내 색상 검출 코드(왼 볼)
                    for (int q = 0; q < cheek.height(); q++) {
                        for (int w = 0; w < cheek.width(); w++) {
                            double[] data = cheek.get(q, w);
                            //Cg 색상 데이터 계산 코드
                            //data[0]=R 색상, data[1]=G 색상, data[2]=B 색상;
                            cg += (-(0.250 * data[0]) + (0.500 * data[1]) - (0.250 * data[2])) + 128;
                        }
                    }

                    //전체 영역에서 더해진 Cg 색상 값을 ROI 영역의 넓이로 나눔
                    cg /= (cheek.height() * cheek.width());
                    Cg_sum += cg;

                    //배열에 색상 데이터 삽입
                    Cg_Left.add(cg);

                    //Cg 색상 데이터 값 초기화
                    cg = 0;
                }
            }

            if (faces.empty()) {
                count += 0;
            } else {
                count += 1;
            }

            //시간 1초씩 내리는 코드
            for(int k = 1; k < 11; k++){
                if(count == 32*k){
                    time = time - 1;
                }

            }

            //기존 12초 설정 10초가 될 경우 측정 시작 알림 제공
        } else if(time > 10){

            MatOfRect faces = new MatOfRect();
            if (mAbsoluteFaceSize == 0) { //사이즈 설정
                long height = mGray.rows();
                if (Math.round(height * mRelativeFaceSize) > 0)
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }

            Mat mGray = new Mat(mRgba.width(), mRgba.height(), CvType.CV_8UC1);
            Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_BGR2GRAY);

            Core.flip(mGray.t(), mGray, 0);

            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

            fx = 0;
            fy = 0;
            fw = 0;
            fh = 0;

            //얼굴 검출 코드
            Rect[] facesArray = faces.toArray();
            for (Rect aFacesArray : facesArray) {
                //평균필터를 이용한 얼굴 검출 안정화
                if (tlx_queue.size() < 15) {
                    tlx_queue.offer(mRgba.width() - aFacesArray.tl().y);
                    tly_queue.offer(aFacesArray.tl().x);
                    brx_queue.offer(mRgba.width() - aFacesArray.br().y);
                    bry_queue.offer(aFacesArray.br().x);
                } else {
                    tlx_queue.offer(mRgba.width() - aFacesArray.tl().y);
                    tly_queue.offer(aFacesArray.tl().x);
                    brx_queue.offer(mRgba.width() - aFacesArray.br().y);
                    bry_queue.offer(aFacesArray.br().x);

                    double tlx = get_avg(tlx_queue.toArray());
                    double tly = get_avg(tly_queue.toArray());
                    double brx = get_avg(brx_queue.toArray());
                    double bry = get_avg(bry_queue.toArray());

                    Imgproc.rectangle(mRgba, new Point(tlx, tly), new Point(brx, bry), FACE_RECT_COLOR, 2);
                    //볼(하) 검출 코드;
                    cx = (int) (((tlx - brx) * 0.35) + brx) - 10;
                    cy = (int) (((bry - tly) * 0.72) + tly);
                    cw = 10;
                    ch = 10;

                    // 볼 확인 코드
//                    Imgproc.rectangle(mRgba, new Point(cx, cy), new Point(cx+cw, cy+ch), CHEEK_RECT_COLOR, 2);
                    Rect roi = new Rect(cx, cy, cw, ch); // 좌측 볼 부분 만큼 이미지 자르기
                    cheek = new Mat(mRgba, roi);

                    //관심 영역 내 색상 검출 코드(왼 볼)
                    for (int q = 0; q < cheek.height(); q++) {
                        for (int w = 0; w < cheek.width(); w++) {
                            double[] data = cheek.get(q, w); //data[0]=r, data[1]=g, data[2]=b;
//                            cg += (-(0.250 * data[0]) + (0.500 * data[1]) - (0.250 * data[2])) + 128;
                        }
                    }
                    cg = 0;
                }
            }

            if (faces.empty()) {
                count += 0;
            } else {
                count += 1;
            }
            if(time <= 12) {
                if (count == 30 || count == 60) {
                    time = time - 1;
                }
            } else if(time <= 18 && time > 12){
                if (count == 30 || count == 60 || count ==90 || count == 120 || count ==150 || count ==180) {
                    time = time - 1;
                    if(count == 120){
                        mHandler.postDelayed(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(PulseActivity.this, "다시 측정하려면, \n 계속바라보세요.", Toast.LENGTH_LONG).show();
                                mp3.start();
                            }
                        }, 0);
                    }
                }
                if(time == 12){
                    count = 0;
                }
            }
        }
        return mRgba;
    }

    @Override
    public void onBackPressed() {
        finish();
    }

    public void fileSave() {
        File file;
        String path = Environment.getExternalStorageDirectory() + "/data";
        file = new File(path);

        if (!file.exists()) {
            file.mkdir();
        }
        file = new File(path + "/Cg.txt");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            String str = "R \n";
            for (int i = 0; i < Cg_Left.size(); i++) {
                str = Double.toString(Cg_Left_arr[i]) + "\n";
                fos.write(str.getBytes());
            }
            str = Double.toString(Cg_Left_arr.length);
            fos.write(str.getBytes());
            fos.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void getBioSignal() {
        //Cg신호로 부터 파워스펙트럼 계산
        //arrLength = N
        tech.goqba.traceeye.pulse.FFT mFFT = new tech.goqba.traceeye.pulse.FFT(512);
        double ai[] = new double[Cg_Left_arr.length];
        double ar[] = new double[Cg_Left_arr.length];
        double power[] = new double[Cg_Left_arr.length];

        for (int i = 0; i < Cg_Left_arr.length; i++) {
            ar[i] = Cg_Left_arr[i];
        }

        mFFT.fft(ar, ai);

        for (int i = 0; i < Cg_Left_arr.length; i++) {
            power[i] = ar[i] * ar[i] + ai[i] * ai[i];
        }

        fps = Cg_Left_arr.length / 10;

        //맥박 계산
        double reg_pulse = 0;

        //맥박 50~120 제한
        pulseRate = (int) ((getPulseRate(power) / 512.0 * fps * 60)) ;

        //맥박이 100 이하 60 이상이면 회귀식 적용
        if(pulseRate <= 100){
            reg_pulse = -0.011 * (pulseRate * pulseRate) + 2.112 * pulseRate - 18.93;
        } else {
            reg_pulse = pulseRate;
        }
        pulseRate = (int) reg_pulse;
    }

    //주파수 파워 분석부
    public double getPulseRate(double[] signal) {
        //50 -> 426.666
        //60 -> 512
        //100 -> 853.3333
        //110 -> 938.6666
        //120 -> 1024

        //60~110 영역 제한 변수
        double x_1 = 512/fps;
        double x_2 = 938.6666/fps;

        //50~120 영역 제한 변수
        //double x_1 = 426.666/fps;
        //double x_2 = 1024/fps;

        double m = x_1;
        double m_2 = x_2;
        int m_3 = (int)m;
        int m_4 = (int)m_2;

        //주파수 최대값 검출
        for (int j = m_3+1; j < m_4+1; j++) {
            if (signal[(int) m] >= signal[j]) {
                m = j;
            }
        }
        //최대 주파수 값 m
        return m;
    }

    public double get_avg(Object[] signal){
        double avg=0;

        for(int i=0;i<signal.length;i++){
            avg += (double)signal[i];
        }
        return (avg/(double)signal.length);
    }

}

class ppgWorkerThread extends Thread {
    Handler handler;
    ppgWorkerThread(Handler handler) {
        this.handler = handler;
    }

    public void run() {
        for (int i = 10; PulseActivity.running ; i--) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
            }
            Message msg = new Message();
            msg.what = 1;
            msg.arg1 = i-1;
            handler.sendMessage(msg);

            if(i == 0)
            {
                msg.what = 2;
                break;
            }
        }
    }
}
