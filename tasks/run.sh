set -e


MULTI_GPU=false
GPU_VISIBLE=7
GPU_NUM=1

CONFIG_PATH=""

ADDR="127.0.0.7"
PORT="29889"

LOG_DIR="./log/"
LOG_FILE="run.log"


if [ ! -d ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
fi

if [ ! -d "output" ]; then
    mkdir "output"
fi


function WriteLog()
{
    local msg_date=`date +%Y-%m-%d" "%H:%M:%S`
    local msg_begin=""
    local msg_end=""
    if [ $# -eq 1 ]; then
        local msg=$1
        echo "[${msg_date}]${msg}" | tee -a ${LOG_DIR}${LOG_FILE}
    elif [ $# -eq 2 ]
    then
        local msg=$2
        local runstat=$1
        if [ ${runstat} -eq 0 ]; then
            msg_begin="Success"
            msg_end="ok!"
        else
            msg_begin="Error"
            msg_end="fail!"
        fi
        echo "[${msg_date}][${msg_begin}]${msg} ${msg_end}" | tee -a ${LOG_DIR}${LOG_FILE}
        if [ ${runstat} -ne 0 ]; then
            echo "error when Task ${msg} runs at ${msg_date}" | tee -a ${LOG_DIR}${LOG_FILE}
            exit 1
        fi
    else
        echo "WriteLog param num should be 1 or 2, actual $#" | tee -a ${LOG_DIR}${LOG_FILE}
        exit 1
    fi
}


function array_join(){
    # 第一个是连接符
    # 后面都是要连接的字符串
    # 逗号连接
    # array_join "," ${array}
    # 空字符连接
    # array_join "" ${array}
    local sep=${1}
    shift 1
    local res=""
    for cur_str in $@
    do
        if [ "${res}"x != ""x ]; then
            res=${res}${sep}
        fi
        res=${res}${cur_str}
    done
    echo ${res}
}

function parse_arguments(){
    WriteLog "arguements: [$*]"
    #-o或--options选项后面是可接受的短选项，如ab:c::，表示可接受的短选项为-a -b -c，
    #其中-a选项不接参数，-b选项后必须接参数，-c选项的参数为可选的
    #-l或--long选项后面是可接受的长选项，用逗号分开，冒号的意义同短选项。
    #-n选项后接选项解析错误时提示的脚本名字
    SHORT_ARG_LIST=(
        "g:"
        "c:"
        "a:"
        "p:"
    )
    LONG_ARG_LIST=(
        "gpu:"
        "config:"
        "addr:"
        "port:"
    )
    SHORT_ARGS=`array_join "" ${SHORT_ARG_LIST[@]}`
    WriteLog "SHORT_ARGS:${SHORT_ARGS}"
    LONG_ARGS=`array_join "," ${LONG_ARG_LIST[@]}`
    WriteLog "LONG_ARGS=${LONG_ARGS}"
    ARGS=`getopt -o ${SHORT_ARGS} --long ${LONG_ARGS} -n "$0" -- "$@"`
    WriteLog $? "parse arguments"
    #将规范化后的命令行参数分配至位置参数（$1,$2,...)
    eval set -- "${ARGS}"
    WriteLog "formatted parameters=[$*]"
    while true
    do
        case "$1" in
            -g|--gpu)
                GPU_VISIBLE="$2"
                GPU_ARRAY=(${GPU_VISIBLE//,/ })
                GPU_NUM=${#GPU_ARRAY[@]}
                shift 2
                ;;
            -c|--config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            -a|--addr)
                ADDR="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                WriteLog "parse error!"
                exit 1
                ;;
        esac
    done
    #剩余的参数
    WriteLog "remaining parameters=[$@]"
}

parse_arguments $@


if [ "${CONFIG_PATH}"x == ""x ]; then
    echo -e "param 'config' is needed."
    exit 1
fi


if [ ${GPU_NUM} -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPU_VISIBLE} \
    python -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} \
        --master_addr ${ADDR} \
        --master_port ${PORT} \
        src/run_with_json.py \
            --param_path ${CONFIG_PATH}
else
    CUDA_VISIBLE_DEVICES=${GPU_VISIBLE} \
    python src/run_with_json.py \
        --param_path ${CONFIG_PATH}
fi

# sh bin/run.sh -g 4,5,6,7 -c examples/gru_classification.json
# sh bin/run.sh -g 7 -c examples/gru_classification.json
