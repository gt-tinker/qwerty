use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebugLoc {
    pub file: String, // Path or file name
    pub line: usize,  // Line number (starting from 1)
    pub col: usize,   // Column number (starting from 1)
}

impl fmt::Display for DebugLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.col)
    }
}
