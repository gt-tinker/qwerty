#include "gtest/gtest.h"
#include "ast.hpp"

TEST(span, fullSpanIsFull) {
    FullSpan fsp(3);
    EXPECT_TRUE(fsp.fullySpans());
}

TEST(span, veclistSpanIsNotFull) {
    VeclistSpan vsp(/*prim_basis=*/Z);
    vsp.vecs.emplace(3, 0b110, false);
    vsp.vecs.emplace(3, 0b101, false);

    EXPECT_FALSE(vsp.fullySpans());
}

TEST(span, veclistSpanIsFull) {
    VeclistSpan vsp(/*prim_basis=*/Z);
    vsp.vecs.emplace(2, 0b10, false);
    vsp.vecs.emplace(2, 0b01, false);
    vsp.vecs.emplace(2, 0b11, false);
    vsp.vecs.emplace(2, 0b00, false);

    EXPECT_TRUE(vsp.fullySpans());
}

//////////////// TRIVIAL CASES ////////////////

// span({'+','-'}) == span(std)
TEST(spanListTrivial, equalityVecStd) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/1, /*isSigned=*/false), // '-'
    }));
    SpanList rhs;
    rhs.append(std::make_unique<FullSpan>(1)); // std[1]

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span({'+'}) != span(std)
TEST(spanListTrivial, equalityPredStd) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));
    SpanList rhs;
    rhs.append(std::make_unique<FullSpan>(1)); // std[1]

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'+'}) != span({'0'})
TEST(spanListTrivial, equalityp0) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));
    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'+'}) != span({'-'})
TEST(spanListTrivial, equalitypm) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));
    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/1, /*isSigned=*/false), // '-'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'+'}) != span({'-','+'})
TEST(spanListTrivial, equalitypmp) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));
    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/1, /*isSigned=*/false), // '-'
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '+'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span(pm) == span(std)
TEST(spanListTrivial, equalityPmStd) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // pm[1]
    SpanList rhs;
    rhs.append(std::make_unique<FullSpan>(1)); // std[1]

    EXPECT_EQ(lhs, rhs);
}

// span(pm[2]) != span(std)
TEST(spanListTrivial, equalityPm2Std) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(2)); // pm[2]
    SpanList rhs;
    rhs.append(std::make_unique<FullSpan>(1)); // std[1]

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

//////////////// SPLITTING FULL SPANS ////////////////

// span(std[2] + std[3]) == span(fourier[5])
TEST(spanListSplitFull, std2Std3Fourier5) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(2));
    lhs.append(std::make_unique<FullSpan>(3));
    SpanList rhs;
    rhs.append(std::make_unique<FullSpan>(5));
    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span(std[2]) == span({'i','j'} + {'0','1'})
TEST(spanListSplitFull, std2VecijVec01) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(2)); // std[2]
    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // 'i'
        llvm::APInt(/*numBits=*/1, /*val=*/1, /*isSigned=*/false), // 'j'
    }));
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0, /*isSigned=*/false), // '0'
        llvm::APInt(/*numBits=*/1, /*val=*/1, /*isSigned=*/false), // '1'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

//////////////// FACTORING VECLISTS ////////////////

// span(std + std) == span({'00','10','01','11'})
TEST(spanListFactorVeclist, stdStdVeclist00100111) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b11, /*isSigned=*/false), // '11'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span(std + {'1'}) == span({'01','11'})
TEST(spanListFactorVeclist, std1Veclist0111) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b11, /*isSigned=*/false), // '11'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span(std + {'1'}) != span({'01','00'})
TEST(spanListFactorVeclist, std1Veclist0100) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span(std + {'1'}) != span({'01','10'})
TEST(spanListFactorVeclist, std1Veclist0110) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '00'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span(std + {'1'}) != span({'01','10','11'})
TEST(spanListFactorVeclist, std1Veclist011011) {
    SpanList lhs;
    lhs.append(std::make_unique<FullSpan>(1)); // std[1]
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
        llvm::APInt(/*numBits=*/2, /*val=*/0b11, /*isSigned=*/false), // '11'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'+','-'} + {'1'}) == span({'01','11'})
TEST(spanListFactorVeclist, veclistpmVeclist0111) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b11, /*isSigned=*/false), // '11'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}


// span({'+','-'} + {'1'}) == span({'j','i'} + {'1'})
TEST(spanListFactorVeclist, veclistpmVeclistji) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // 'j'
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // 'i'
    }));
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span({'+','-'} + {'i'}) == span({'ji','ii'}
TEST(spanListFactorVeclist, veclistpmiVeclistjii) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // 'i'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // 'ji'
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // 'ii'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span({'+','-'} + {'+'}) != span({'ji','ii'}
TEST(spanListFactorVeclist, veclistpmpVeclistjii) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
    }));
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // 'ji'
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // 'ii'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'+','-'} + {'j','i'}) == span({'00','01','10','11'})
TEST(spanListFactorVeclist, veclistpmjiVeclist00011011) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Y, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // 'j'
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // 'i'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
        llvm::APInt(/*numBits=*/2, /*val=*/0b11, /*isSigned=*/false), // '11'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span({'0', '1'} + {'0'}) != span({'001'})
TEST(spanListFactorVeclist, veclist000Veclist001) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '0'
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '0'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/3, /*val=*/0b001, /*isSigned=*/false), // '001'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

//////////////// FACTORING VECLISTS OUT OF VECLISTS ////////////////

// span({'+'} + {'-','+'}) == span({'++','+-'})
TEST(spanListFactorVecFromVec, veclistPmpVeclistpppm) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
    }));
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '++'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '+-'
    }));

    EXPECT_EQ(lhs, rhs);
    EXPECT_EQ(rhs, lhs);
}

// span({'0'} + {'-','+'}) != span({'++','+-'})
TEST(spanListFactorVecFromVec, veclist0mpVeclistpppm) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
    }));
    lhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '-'
        llvm::APInt(/*numBits=*/1, /*val=*/0b0, /*isSigned=*/false), // '+'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(X, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '++'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '+-'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'00', '01', '10'} + {'1'}) != span({'001','011','101','111'})
TEST(spanListFactorVecFromVec, veclist0001101Veclist001011101111) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/3, /*val=*/0b001, /*isSigned=*/false), // '001'
        llvm::APInt(/*numBits=*/3, /*val=*/0b011, /*isSigned=*/false), // '011'
        llvm::APInt(/*numBits=*/3, /*val=*/0b101, /*isSigned=*/false), // '101'
        llvm::APInt(/*numBits=*/3, /*val=*/0b111, /*isSigned=*/false), // '111'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'00', '01', '10'} + {'1'}) != span({'001','011','111'})
TEST(spanListFactorVecFromVec, veclist0001101Veclist001011111) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/3, /*val=*/0b001, /*isSigned=*/false), // '001'
        llvm::APInt(/*numBits=*/3, /*val=*/0b011, /*isSigned=*/false), // '011'
        llvm::APInt(/*numBits=*/3, /*val=*/0b111, /*isSigned=*/false), // '111'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'00', '01', '10'} + {'1'}) != span({'001','011','010'})
TEST(spanListFactorVecFromVec, veclist0001101Veclist001011010) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/3, /*val=*/0b001, /*isSigned=*/false), // '001'
        llvm::APInt(/*numBits=*/3, /*val=*/0b011, /*isSigned=*/false), // '011'
        llvm::APInt(/*numBits=*/3, /*val=*/0b010, /*isSigned=*/false), // '010'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}

// span({'00', '01', '10'} + {'1'}) != span({'001','011','100'})
TEST(spanListFactorVecFromVec, veclist0001101Veclist001011100) {
    SpanList lhs;
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/2, /*val=*/0b00, /*isSigned=*/false), // '00'
        llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
        llvm::APInt(/*numBits=*/2, /*val=*/0b10, /*isSigned=*/false), // '10'
    }));
    lhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '1'
    }));

    SpanList rhs;
    rhs.append(std::make_unique<VeclistSpan>(Z, std::initializer_list<llvm::APInt>{
        llvm::APInt(/*numBits=*/3, /*val=*/0b001, /*isSigned=*/false), // '001'
        llvm::APInt(/*numBits=*/3, /*val=*/0b011, /*isSigned=*/false), // '011'
        llvm::APInt(/*numBits=*/3, /*val=*/0b100, /*isSigned=*/false), // '100'
    }));

    EXPECT_NE(lhs, rhs);
    EXPECT_NE(rhs, lhs);
}
