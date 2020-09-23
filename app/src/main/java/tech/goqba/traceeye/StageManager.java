package tech.goqba.traceeye;

import android.content.Context;

import tech.goqba.traceeye.Gaze.EyesTracker;
import tech.goqba.traceeye.androidDraw.AbstractRenderView;
import tech.goqba.traceeye.androidDraw.AdjustView;
import tech.goqba.traceeye.androidDraw.StageReport;
import tech.goqba.traceeye.androidDraw.StageView1;
import tech.goqba.traceeye.androidDraw.StageView2;
import tech.goqba.traceeye.androidDraw.StageView3;
import tech.goqba.traceeye.androidDraw.StageView4;

public class StageManager implements EyesTracker.Callback {
    public static final int STAGE1 = 0;
    public static final int STAGE2 = 1;
    public static final int STAGE3 = 2;
    public static final int STAGE4 = 3;
    public static final int ADJUST = 4;
    public static final int STAGE_START_COUNT = 5;
    public static final int STAGE_REPORT = 6;
    private Context mContext;
    private AbstractRenderView.ViewCallback mCallback;
    private AbstractRenderView mCurrentView = null;

    public StageManager(Context context, AbstractRenderView.ViewCallback callback) {
        mContext = context;
        mCallback = callback;
    }

    public AbstractRenderView getStage(int index) {
        mCurrentView = getStageImpl(index);
        return mCurrentView;
    }

    public void draw(int x, int y) {
        if (mCurrentView != null) {
            mCurrentView.draw(x, y);
        }
    }

    public void onOrientationChange() {
        mCurrentView.orientChange();
    }

    public AbstractRenderView getStageImpl(int index) {
        switch (index) {
            case STAGE1:
                return new StageView1(mContext, mCallback);
            case STAGE2:
                return new StageView2(mContext, mCallback);
            case STAGE3:
                return new StageView3(mContext, mCallback);
            case STAGE4:
                return new StageView4(mContext, mCallback);
            case STAGE_REPORT:
                return new StageReport(mContext, mCallback);
            case ADJUST:
                return new AdjustView(mContext, mCallback);
        }
        return null;
    }

    @Override
    public void onChangePosition(int x, int y) {
        draw(x,y);
    }

    @Override
    public void onStartTracker() {

    }

    @Override
    public void onCalibrationProgress(float progress) {
        mCurrentView.onCalibrationProgress(progress);
    }

    @Override
    public void onCalibrationNextPoint(float x, float y) {
        mCurrentView.onCalibrationNextPoint(x,y);
    }

    @Override
    public void onCalibrationFinished() {
        mCurrentView.onCalibrationFinished();
    }
}