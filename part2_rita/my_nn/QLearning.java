package my_nn;

import javafx.util.Pair;
import java.util.Arrays;

public class QLearning {
    int N_STATES = 20;// 1维世界的宽度，6个格子[0][1][2][3][4][5]
    String[] ACTIONS = {"left", "right"};// 探索者的可用动作
    double ALPHA = 0.1;// 学习率
    double GAMMA = 0.9;// 可视奖励递减值
    double EPSILON = 0.2;// 贪婪度greedy，90%的概率采取当前最优选择
    int MAX_EPISODES = 20;// 最大回合数
    double FRESH_TIME = 0.1;// 移动间隔时间,s
    boolean offPolicy = false;

    public static void main(String[] args) {
        QLearning my_rl = new QLearning();
        double[][] q_table = my_rl.RL();
        System.out.print("\rQ-table:");
        System.out.println("  left  right");// 打印强化学习后更新出来的Q表
        int i = 0;
        for (double[] row:q_table) {
            System.out.print(i + " ");
            for (double num:row) {
                System.out.printf("%.6f ", num);
            }
            ++i;
            System.out.println();
        }
    }

    double[][] build_q_table(int n_states, String[] actions) {// 建立一个价值表，Q表
        return new double[n_states - 1][actions.length];// n_states-1终点位置不需要Q值
        // DataFrame建表函数，建立一个值全为0的表，其中列为动作，行为每一state
        // table[0][0] = 0.000552
        // table[0][1] = 0.029149
        // table[1][0] = 0.000000
        // table[1][1] = 0.100716
        // table[2][0] = 0.000030
        // table[2][1] = 0.272388
        // table[3][0] = 0.000000
        // table[3][1] = 0.560801
        // table[4][0] = 0.027621
        // table[4][1] = 0.878423
    }

   int choose_action(int state, double[][] q_table) {// 选择动作
        double[] state_actions = q_table[state];
        // 以state表示的数字的那一行的所有数据
        int action;
        if ((Math.random() <= EPSILON) || (max(state_actions) == 0)){
            // uniform(x,y，z)从均匀分布[x,y)中随机采样z个数据，默认值x=0，y=1，z=1;max最大值是0，则全为0
            // 从[0,1)中随机产生一个数，如果这个数大于EPSILON(0.9)，或者如果Q表中state行两个数都等于0，那么执行下一步
            action = random_choice(ACTIONS);// 从['left', 'right']中随机选择一个动作执行
        } else {
            action = argmax(state_actions);// 从Q表的state行里选择出最大值，得到它的索引left或者right
        }
        return action;
    }

    double max(double[] arr) {
        double max = arr[0];
        for (double num:arr) {
            if (num > max) {
                max = num;
            }
        }
        return max;
    }

    int argmax(double[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] > arr[max]) {
                max = i;
            }
        }
        return max;
    }

    int random_choice(String[] actions) {
        double ran = Math.random();
        int i = 0;
        for (; i < actions.length; ++i) {
            if (ran >= (double)i / actions.length && ran < (double)(i + 1) / actions.length) {
                break;
            }
        }
        return i;
    }

    Pair<Integer, Double> get_env_feedback (int S, int A){// 环境反馈
        int S_;
        double Reward;
        if (A == 1) {// 如果动作是向右
            if (S == N_STATES - 2) {// 6-2=4
                // 总共6个格子[0][1][2][3][4][5]，站在[4]上，下一个动作已经决定向右，则肯定到达终点
                S_ = N_STATES - 1;// 下一个动作是终点，结束
                Reward = 1;// 奖励为1
            }
        else {
                S_ = S + 1;// 站在下一个格子上
                Reward = 0;// 奖励为0
            }
        }
    else {// 如果动作是向左
            Reward = 0;// 奖励为0，走左边永远到不了终点，奖励必须为0
            if (S == 0) {// 如果站在[0]上，已经选择向左
                S_ = S;// 那么下一位置不变，仍站在[0]
            }
            else {
                S_ = S - 1;// 其他位置上则向左挪一格
            }
        }
        return new Pair<>(S_, Reward);
    }

    void update_env(int S, int episode, int step_counter) {// 环境更新
        char[] env = new char[N_STATES];
        Arrays.fill(env, '-');
        env[N_STATES - 1] = 'T';
        // 一维世界是由[0][1][2][3][4]的'-'和一个'T'组成
        if (S == N_STATES - 1) {// 如果站在终点上
            String a = "Episode " + (episode + 1) + ": total_steps = " + step_counter;
            System.out.println("" + a);// 将光标移到本行的开头，重新打印输出,end=''不换行
            try {
                Thread.sleep(1000);// 等待两秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            // System.out.print("\r                                ");// 用空白刷新
        } else {// 如果没在终点
            env[S] = 'o';// 探索者站在哪个格子上，哪个格子变成'o'
            String a = new String(env);
//            System.out.println("" + a);// 打印一维世界
            try {
                Thread.sleep((long) FRESH_TIME * 1000);// 等待移动间隔时间
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private double[][] RL() {
        double[][] q_table = build_q_table(N_STATES, ACTIONS);// 实例化Q表
        int total_steps = 0;
        for(int episode = 0; episode < MAX_EPISODES; ++episode) {// 在总回合数以内，每一回合
            int step_counter = 0;// 总步数计数器
            int S = 0;// 初始时站在[0]上
            boolean is_terminated = false;// 是否结束
            update_env(S, episode, step_counter);// 初始环境
            double q_target;
            while(!is_terminated) {// 没有结束时，每一步，每次结束返回for循环
                int A = choose_action(S, q_table);// 选择当前动作
                Pair<Integer, Double> feedback = get_env_feedback(S, A);// 获得行动反馈
                int S_ = feedback.getKey();
                double Reward = feedback.getValue();
                double q_predict = q_table[S][A];// 获取当前位置当前动作的Q值
                // loc基于标签比如，loc[2,right]：第二行right列
                if(S_ != N_STATES - 1) {// 如果下一步没有结束
                    if (offPolicy) {
                        q_target = Reward + GAMMA * max(q_table[S_]);// Q Learning关键公式1
                    } else {
                        q_target = Reward + GAMMA * q_table[S][A];
                        // 下一步那一行中的最大值：下一步中最有价值的行动
                    }
                }
                else {// 下一步就结束
                    q_target = Reward;// 可视奖励为Reward，1
                    is_terminated = true;// 结束，不再进入循环
                }
                q_table[S][A] += ALPHA * (q_target - q_predict);// Q Learning关键公式2
                // 用学习率乘上目标与实际获得Q值的差值来更新当前位置当前动作的Q值
                S = S_;// 站到下一步的位置
                update_env(S, episode, step_counter + 1);// 没有结束时，每一步都更新环境
                step_counter += 1;// 总步数计数器+1
            }
            total_steps += step_counter;
        }
        System.out.print("\raverage step: " + total_steps / MAX_EPISODES);
        return q_table;// 更新Q表
    }
}
