#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebugLoc {
    pub file: String, // Path or file name
    pub line: usize,  // Line number (starting from 1)
    pub col: usize,   // Column number (starting from 1)
}
