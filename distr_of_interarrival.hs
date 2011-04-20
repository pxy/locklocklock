
import Data.Ord
import Data.List
import System.IO
import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as PT
import Text.ParserCombinators.Parsec.Language (emptyDef)


-- to be recycled
main2 =do dat <- getContents
          putStr . unwords . intersperse "\n" $ map show $ interDist ((map read $ words dat) :: [Double])


-- to be recycled
main3 = do dat <- getContents
           parsed <- myparse row dat
           putStrLn $ show $ (waittime (map fst (sortBy (comparing snd) parsed)) / ((fromIntegral (length parsed)) / 2))
           return ()


main = do dat <- getContents
          parsed <- myparse row dat -- already sorted
          return ()


-- lock sequence graph from a threads perspective
-- get all the lock accesses sorted by time
-- for each lock, get the list of (non-distinct) locks following
-- calculate percentage
-- generate distinct lists

filterByTIDAndSortByTime tid list = sortBy (comparing fst) $ filter ((== tid) . snd) list


{- ******** LIST FUNCTIONS ******** -}

-- | a new list with the inter element distances in l
interDist :: (Num a) => [a] -> [a]
interDist l = zipWith (-) (tail l) l

-- | sum of all inter element distances
-- PRE: list contains even number of elements
waittime :: (Num a) => [a] -> a
waittime [] = 0
waittime (x1:x2:xs) = x2 - x1 + waittime xs


{- ******* PARSING FUNCTIONS ******** -}


-- | parse it. 
-- takes a sequence of records that f should be able to parse as input
myparse f input = case parse (file f) "" input of
	            Left er  -> return []
	            Right cl -> return $ cl

	 
-- different record types

row :: CharParser st (Double, Integer)
row = do (char 'r' <|> char 't') >> space
         x1 <- double
         x2 <- integer
         return (x1, x2)

row2 :: CharParser st (Integer, Integer)
row2 = do (char 'r' <|> char 't') >> space
          x1 <- integer
          x2 <- integer
          return (x1, x2)

row3 :: CharParser st (Double, Integer, Integer)
row3 = do (char 'r' <|> char 't') >> space
          x1 <- double
          x2 <- integer
          x3 <- integer
          return (x1, x2, x3)


{- ok, just some crappy renaming. one file is many rows -}
file :: CharParser st a -> CharParser st [a]
file = many 

-- | function generating a token parser based on a
-- lexical parsers combined with a language record definition
lexer :: PT.TokenParser st
lexer  = PT.makeTokenParser emptyDef                                  
 
integer :: CharParser st Integer
integer = PT.integer lexer                                            
 
double :: CharParser st Double
double = PT.float lexer

