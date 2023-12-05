pub fn batchify<T: Clone>(input: Vec<T>, batch_size: usize) -> Vec<Vec<T>> {
    let l = input.len();
    let mut output = Vec::new();

    for i in (0..l).step_by(batch_size) {
        let end = std::cmp::min(i + batch_size, l);
        let batch = input[i..end].to_vec();
        output.push(batch);
    }

    output
}