use tch::Tensor;

pub fn get_len_util_stop(sequence: &Tensor, end_index: usize) -> usize {
    for (i, val) in sequence.iter::<i64>().unwrap().enumerate() {
        if val == end_index as i64 {
            return i + 1
        }
    }

    sequence.size()[0] as usize
}